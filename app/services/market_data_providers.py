"""Provider clients for market-watch signals (weather, events, holidays)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class WeatherForecastDay:
    """Normalized one-day weather forecast used by market-watch analysis."""

    day: date
    weather_code: int | None
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_mm: float | None
    snowfall_cm: float | None
    wind_kph_max: float | None


@dataclass
class NearbyEvent:
    """Normalized nearby event record used by impact scoring."""

    name: str
    source_url: str | None
    start_at_utc: datetime | None
    venue_name: str | None
    latitude: float | None
    longitude: float | None
    distance_km: float | None
    category: str | None
    popularity_hint: str | None


@dataclass
class PublicHoliday:
    """Normalized holiday signal entry."""

    day: date
    local_name: str
    name: str


class MarketDataProviders:
    """Thin API client wrappers with normalized outputs and defensive parsing."""

    def __init__(
        self,
        *,
        ticketmaster_api_key: str | None,
        timeout_seconds: int = 20,
    ) -> None:
        self.ticketmaster_api_key = ticketmaster_api_key
        self.timeout_seconds = max(5, timeout_seconds)

    def fetch_weather_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        lookahead_days: int,
    ) -> tuple[list[WeatherForecastDay], dict[str, Any]]:
        """Fetch daily forecast from Open-Meteo and normalize relevant fields."""

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "snowfall_sum",
                    # Open-Meteo changed naming over time; both are parsed downstream.
                    "wind_speed_10m_max",
                    "windspeed_10m_max",
                ]
            ),
            "timezone": "UTC",
            "forecast_days": max(1, int(lookahead_days)),
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"
        try:
            payload = self._get_json(url)
        except Exception as exc:
            return [], {"status": "error", "provider": "open-meteo", "error": f"{type(exc).__name__}: {exc}"}

        daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
        days = daily.get("time", [])
        weather_codes = daily.get("weather_code", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precipitation = daily.get("precipitation_sum", [])
        snowfall = daily.get("snowfall_sum", [])
        wind = daily.get("wind_speed_10m_max", daily.get("windspeed_10m_max", []))

        out: list[WeatherForecastDay] = []
        for idx, day_value in enumerate(days):
            try:
                out.append(
                    WeatherForecastDay(
                        day=date.fromisoformat(str(day_value)),
                        weather_code=self._safe_int(self._value_at(weather_codes, idx)),
                        temp_max_c=self._safe_float(self._value_at(temp_max, idx)),
                        temp_min_c=self._safe_float(self._value_at(temp_min, idx)),
                        precipitation_mm=self._safe_float(self._value_at(precipitation, idx)),
                        snowfall_cm=self._safe_float(self._value_at(snowfall, idx)),
                        wind_kph_max=self._safe_float(self._value_at(wind, idx)),
                    )
                )
            except Exception:
                # Skip malformed rows instead of failing the whole provider pass.
                continue

        return out, {"status": "ok", "provider": "open-meteo", "count": len(out)}

    def fetch_ticketmaster_events(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: int,
        start_at_utc: datetime,
        end_at_utc: datetime,
    ) -> tuple[list[NearbyEvent], dict[str, Any]]:
        """Fetch nearby events from Ticketmaster Discovery API."""

        if not self.ticketmaster_api_key:
            return [], {"status": "skipped", "provider": "ticketmaster", "reason": "missing_api_key"}

        params = {
            "apikey": self.ticketmaster_api_key,
            "latlong": f"{latitude},{longitude}",
            "radius": max(1, int(radius_km)),
            "unit": "km",
            "startDateTime": start_at_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "endDateTime": end_at_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "sort": "date,asc",
            "size": 100,
        }
        url = f"https://app.ticketmaster.com/discovery/v2/events.json?{urlencode(params)}"
        try:
            payload = self._get_json(url)
        except Exception as exc:
            return [], {"status": "error", "provider": "ticketmaster", "error": f"{type(exc).__name__}: {exc}"}

        events_node = (
            payload.get("_embedded", {}).get("events", [])
            if isinstance(payload, dict)
            else []
        )
        out: list[NearbyEvent] = []
        for raw in events_node:
            if not isinstance(raw, dict):
                continue

            venue = self._first(raw.get("_embedded", {}).get("venues", []))
            classifications = raw.get("classifications", [])
            first_class = self._first(classifications) if isinstance(classifications, list) else None
            segment = (
                first_class.get("segment", {}).get("name")
                if isinstance(first_class, dict)
                else None
            )
            genre = (
                first_class.get("genre", {}).get("name")
                if isinstance(first_class, dict)
                else None
            )
            category = str(genre or segment or "").strip() or None

            start_str = raw.get("dates", {}).get("start", {}).get("dateTime")
            start_at = self._parse_utc(start_str)
            venue_lat = self._safe_float(venue.get("location", {}).get("latitude")) if isinstance(venue, dict) else None
            venue_lon = self._safe_float(venue.get("location", {}).get("longitude")) if isinstance(venue, dict) else None

            out.append(
                NearbyEvent(
                    name=str(raw.get("name", "Unnamed event")),
                    source_url=self._first(raw.get("url")) if isinstance(raw.get("url"), list) else raw.get("url"),
                    start_at_utc=start_at,
                    venue_name=venue.get("name") if isinstance(venue, dict) else None,
                    latitude=venue_lat,
                    longitude=venue_lon,
                    distance_km=None,
                    category=category,
                    popularity_hint=self._popularity_hint(raw),
                )
            )

        return out, {"status": "ok", "provider": "ticketmaster", "count": len(out)}

    def fetch_us_public_holidays(
        self,
        *,
        year: int,
    ) -> tuple[list[PublicHoliday], dict[str, Any]]:
        """Fetch US public holidays for one year from Nager.Date (free endpoint)."""

        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
        try:
            payload = self._get_json(url)
        except Exception as exc:
            return [], {"status": "error", "provider": "nager-date", "error": f"{type(exc).__name__}: {exc}"}

        if not isinstance(payload, list):
            return [], {"status": "error", "provider": "nager-date", "error": "unexpected_payload_type"}

        out: list[PublicHoliday] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            day = raw.get("date")
            try:
                parsed = date.fromisoformat(str(day))
            except Exception:
                continue
            out.append(
                PublicHoliday(
                    day=parsed,
                    local_name=str(raw.get("localName", "")),
                    name=str(raw.get("name", "")),
                )
            )
        return out, {"status": "ok", "provider": "nager-date", "count": len(out)}

    def _get_json(self, url: str) -> Any:
        """Perform one JSON GET request with a stable user-agent."""

        request = Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": "airbnb-business-agent/market-watch",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read().decode(charset, errors="replace")
            return json.loads(raw)

    def _value_at(self, values: list[Any], index: int) -> Any:
        """Safe list indexing helper that returns None for out-of-range access."""

        if index < 0 or index >= len(values):
            return None
        return values[index]

    def _safe_float(self, value: Any) -> float | None:
        """Convert to float when possible, otherwise return None."""

        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _safe_int(self, value: Any) -> int | None:
        """Convert to int when possible, otherwise return None."""

        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None

    def _parse_utc(self, value: Any) -> datetime | None:
        """Parse UTC-like timestamps from providers into timezone-aware datetime."""

        if not value:
            return None
        text = str(value).strip()
        try:
            if text.endswith("Z"):
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return None

    def _first(self, value: Any) -> Any:
        """Return first element when the value is a list, otherwise value itself."""

        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _popularity_hint(self, event_payload: dict[str, Any]) -> str | None:
        """Infer a weak popularity hint from available event metadata."""

        classifications = event_payload.get("classifications", [])
        if isinstance(classifications, list):
            first_class = self._first(classifications)
            if isinstance(first_class, dict):
                segment = str(first_class.get("segment", {}).get("name", "")).strip().lower()
                if segment in {"music", "sports"}:
                    return "high"

        ticket_limit = event_payload.get("ticketLimit")
        if ticket_limit is not None:
            return "medium"
        return None


def utc_now() -> datetime:
    """Small utility used by market-watch components for consistent UTC timestamps."""

    return datetime.now(tz=UTC)


def within_days(window_start: datetime, day: date, max_days: int) -> bool:
    """Return True when date falls inside the configured lookahead window."""

    delta = day - window_start.date()
    return timedelta(days=0) <= delta <= timedelta(days=max(0, max_days))
