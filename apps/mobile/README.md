# Book Quiz Mobile

Expo React Native client for practicing the shared quiz question bank on iOS and Android.

## Run

From the repository root:

```bash
npm run mobile:start
```

Set the Supabase values to enable profile-backed rating/report sync without depending on a PC-hosted API server.

```bash
$env:EXPO_PUBLIC_SUPABASE_URL="https://your-project-ref.supabase.co"
$env:EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
npm run mobile:start
```

`EXPO_PUBLIC_QUIZ_API_BASE_URL` is optional and is only used for detailed AI explanation chat. Without Supabase env vars or a signed-in profile, bundled-question practice, local rating updates, and queued reports still work on-device.
