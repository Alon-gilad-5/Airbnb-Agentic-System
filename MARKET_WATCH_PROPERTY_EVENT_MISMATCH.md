# Market Watch Property/Event Mismatch

## Summary

The Market Watch page is currently configured for property `42409434` (`The Burlington Hotel`), but stored Market Watch alerts exist for property `290761`.

Because the UI reads alerts from the persisted Market Watch alert store, not directly from live Ticketmaster/Open-Meteo responses, this mismatch causes the page to show no alerts for the currently selected property even though event alerts exist in storage for a different property.

This is a data-scoping problem, not a Ticketmaster parsing problem.

## What Was Observed

### New UI observation

An additional observation from manual usage changes the interpretation:

- selecting a property on the Reviews screen carries over into Market Watch
- entering the Market Watch console triggers an automatic refresh
- one selected property shows `41` alerts
- the other selected property shows `17` alerts

This is highly relevant because it shows the end-user flow is not completely stuck on one wrong property ID.

It suggests:

- the property selection handoff from Reviews to Market Watch is working
- the Market Watch page is likely querying per selected property
- at least in the environment where this was observed, both configured properties have persisted Market Watch alerts

So the earlier mismatch diagnosis should be treated as environment-specific evidence, not as proof that the production-visible UI is always reading the wrong property.

### Current configured property profiles

From `.env`:

- `ACTIVE_PROPERTY_ID=42409434`
- `ACTIVE_PROPERTY_NAME=The Burlington Hotel`
- `SECONDARY_PROPERTY_ID=10046908`

The UI exposes these through `/api/property_profiles`.

### Current Market Watch API behavior

Scoped to the active property:

```text
GET /api/market_watch/alerts?limit=5&owner_id=owner_001&property_id=42409434
=> {"status":"ok","error":null,"alerts":[]}
```

Scoped to the old property:

```text
GET /api/market_watch/alerts?limit=5&owner_id=owner_001&property_id=290761
=> returns multiple event alerts
```

Example returned alert for `290761`:

- `title`: `Nearby medium-impact event: Los Angeles Lakers vs. Orlando Magic`
- `start_at_utc`: `2026-02-25T03:30:00+00:00`
- `evidence.property_id`: `290761`

### Local SQLite evidence

The local file `data/market_watch_alerts.db` contains persisted Market Watch rows only for property `290761`.

Observed aggregate:

```text
property_id=290761
first_seen=2026-02-15T17:30:40.272172+00:00
last_seen=2026-02-16T10:36:23.710572+00:00
count=13
```

### Important backend note

The running API likely uses a different backend than the local SQLite file for alerts:

- `app/main.py` loads `.env` via `load_dotenv()`
- `.env` sets `DATABASE_URL`
- `create_market_alert_store()` selects Postgres when `DATABASE_URL` is present

This matters because:

- the local SQLite file shows old `290761` rows
- the live API also returns `290761` rows, but with newer timestamps than the local SQLite file

That strongly suggests the running server is reading persisted Market Watch alerts from Postgres, and that backend also contains alerts for `290761`.

### Interpretation after the new UI evidence

The `41` / `17` alert counts imply that the user-visible system is probably operating against a backend that contains valid per-property alert history for the current configured profiles.

That means there are likely multiple realities involved during debugging:

1. historical persisted alerts exist for legacy property `290761`
2. the backend visible in the user workflow also contains alerts for the currently selected configured properties

So the existence of `290761` data is real, but it may not be the direct cause of the current UI behavior the user is seeing.

## Why The DB Is Involved

The Market Watch UI does not render directly from provider API responses.

Actual flow:

1. The page triggers `POST /api/market_watch/run`.
2. The Market Watch agent calls external providers such as Ticketmaster and weather sources.
3. The agent converts the results into alert records.
4. The alert records are persisted into the Market Watch alert store.
5. The page then calls `GET /api/market_watch/alerts`.
6. The UI renders the stored alerts returned by that endpoint.

So the alert store is effectively the Market Watch inbox/history layer.

This means a property mismatch in stored alerts is enough to break what the page shows, even if the provider logic itself is correct.

## Relevant Code Paths

### Property profile source

`app/main.py`

- `_build_property_profiles()` builds UI-selectable property profiles from config.
- `_build_property_profiles_response()` exposes those profiles via `/api/property_profiles`.

### Market Watch run context

`app/main.py`

- `_build_autonomous_context()` starts from the active owner/property in config.
- `_merge_market_watch_context()` overlays request payload values on top of that config.
- `market_watch_run()` executes the Market Watch agent with that merged context.

### Alert persistence

`app/agents/market_watch_agent.py`

- `_persist_alerts_stage()` writes alerts using `base_context["property_id"]`.
- The persisted record ID is also derived from `property_id`.

This is important because it means the app does not internally rewrite `42409434` to `290761`.

### Alert retrieval

`app/main.py`

- `market_watch_alerts()` reads alerts scoped by `owner_id` and `property_id`.

`app/services/market_alert_store.py`

- `list_latest_alerts()` applies `WHERE owner_id = ... AND property_id = ...`.

So if alerts were persisted under `290761`, querying for `42409434` will correctly return zero rows.

## Revised Root Cause Assessment

There is no code path currently converting the configured property `42409434` into `290761`.

The most likely cause is historical persisted data:

- at some earlier point, the app was run with property `290761`, or
- someone manually triggered Market Watch with payload `property_id=290761`

Those alerts were persisted under `290761`, and they still exist in the alert store.

Later, the configured active property was changed to `42409434`, but the stored Market Watch history was not reset or regenerated for the new property.

That explains the stale `290761` data.

However, the later observation that one selected property shows `41` alerts and another shows `17` alerts means the active end-user flow is likely doing the right thing with property selection.

So the best current assessment is:

- legacy/stale persisted alerts for `290761` definitely exist
- the Reviews -> Market Watch property handoff appears to work
- the visible Market Watch system likely has valid alerts for the current configured properties
- therefore the `290761` mismatch is probably a backend-history issue, not the primary explanation for every current Market Watch symptom

Result of the revised assessment:

- current UI profile = `42409434`
- stored Market Watch alerts = `290761`
- page query for `42409434` returns no rows

This result was observed in one queried environment, but the `41` / `17` evidence shows that another effective runtime path or backend state contains property-scoped alerts for the active configured profiles.

## Supporting Context

The repository itself still references `290761` as a historical sample property:

`README.md`

- High-review sample: `42409434`
- Low-review sample: `290761`

This supports the conclusion that `290761` is legacy sample data rather than a value currently derived from the active profile.

## Impact

### User-facing impact

- Market Watch may appear empty for the active property in some environments or backend states.
- The events calendar can show no highlighted dates when the selected property has no matching persisted alerts.
- Debugging becomes confusing because valid event alerts exist in storage, but not for the currently selected property.
- Different environments or stores may appear to contradict each other.

### Operational impact

- Alert history becomes hard to trust unless property scope is checked.
- Old sample/demo data can leak into current workflows.
- Manual testing can produce false conclusions about whether provider fetching is broken.

## Optional Solutions

### Option 1: Clean the old Market Watch alerts

Delete persisted alerts for property `290761` from the active Market Watch alert store.

Pros:

- Fastest cleanup
- Removes stale sample data
- Makes the current UI state less confusing

Cons:

- Loses historical alert records for that property
- Does not prevent the same problem from happening again

Best when:

- `290761` is definitely obsolete
- old Market Watch history is not needed

### Option 2: Regenerate alerts for the current active property

Run Market Watch for `42409434` with working provider connectivity and persist fresh alerts.

Pros:

- Aligns the UI with the current property
- Keeps Market Watch behavior unchanged

Cons:

- Depends on provider/network availability
- Does not remove old mismatched rows unless cleaned separately

Best when:

- the current property configuration is already correct
- you want fresh real alerts instead of only cleanup

### Option 3: Add an admin reset endpoint or script

Add a targeted maintenance action such as:

- delete alerts by `property_id`
- delete alerts older than a threshold
- clear all Market Watch history in non-production/local environments

Pros:

- Reusable operational fix
- Good for demos and local testing

Cons:

- Additional code surface
- Needs permission/guardrails in shared environments

Best when:

- multiple properties are tested regularly
- stale data cleanup will happen more than once

### Option 4: Warn in the UI when the store contains alerts for unknown properties

Add a diagnostic warning when:

- current selected property has zero alerts, but
- the store contains alerts for other property IDs under the same owner

Pros:

- Easier debugging
- Prevents silent confusion

Cons:

- Does not fix the underlying data mismatch
- Requires an extra diagnostic API or query

Best when:

- developer visibility matters
- you want safer debugging without changing backend behavior

### Option 5: Namespace or isolate demo/sample data

Separate sample/local/demo Market Watch history from real active-property history.

Examples:

- separate DB/table/schema for local demo runs
- separate owner IDs for sample data
- environment-specific alert stores

Pros:

- Prevents cross-contamination between sample and real data
- Cleaner long-term architecture

Cons:

- More setup complexity
- Requires migration/operational discipline

Best when:

- the project continues to use sample properties for demos, tests, or experimentation

## Recommended Path

For the current issue, the most practical sequence is:

1. Confirm which backend the browser-visible Market Watch page is using.
2. Compare alert counts for `42409434`, `10046908`, and `290761` in that exact backend.
3. Treat `290761` as legacy data unless the UI can actually still select or generate alerts for it.
4. Delete or archive stale `290761` alerts only after confirming they are not part of an active workflow.
5. Optionally add a small admin cleanup tool or diagnostic warning so mismatched/stale property data is visible next time.

## Final Conclusion

The strongest confirmed fact is that persisted Market Watch alerts exist for a legacy property ID (`290761`).

But the later observation that one selected property shows `41` alerts and another shows `17` alerts indicates that the user-facing Reviews -> Market Watch property flow is probably functioning correctly in at least one runtime environment.

So the most accurate conclusion is:

- stale `290761` Market Watch data exists and should be treated as legacy persisted history
- the UI likely does switch correctly between current properties
- the remaining debugging question is not simply "wrong property is always used"
- the real problem is likely a mixture of stale history plus environment/backend differences
