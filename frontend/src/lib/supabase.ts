import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

if (!url || !key) {
  // Surfaced early in dev so a missing .env.local is obvious.
  console.warn(
    "Supabase env not set: define NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY in frontend/.env.local",
  );
}

// Browser client. Sessions persist in localStorage and refresh automatically;
// the access token is attached to API calls by lib/api.ts.
export const supabase = createClient(url ?? "", key ?? "", {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
  },
});
