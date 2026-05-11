import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  createClient,
  type SupabaseClient,
  type User,
} from "@supabase/supabase-js";

export type MobileProfile = {
  id: string;
  email: string | null;
  displayName: string | null;
};

let client: SupabaseClient | null = null;

export function getConfiguredSupabaseEnv(): {
  url: string;
  publishableKey: string;
} | null {
  const url =
    process.env.EXPO_PUBLIC_SUPABASE_URL?.trim() ??
    process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
  const publishableKey =
    process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY?.trim() ??
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY?.trim();

  if (!url || !publishableKey) return null;
  return { url, publishableKey };
}

export function getMobileSupabaseClient(): SupabaseClient | null {
  if (client) return client;

  const env = getConfiguredSupabaseEnv();
  if (!env) return null;

  client = createClient(env.url, env.publishableKey, {
    auth: {
      storage: AsyncStorage,
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: false,
    },
  });
  return client;
}

export function toMobileProfile(user: User): MobileProfile {
  return {
    id: user.id,
    email: user.email ?? null,
    displayName:
      typeof user.user_metadata?.display_name === "string"
        ? user.user_metadata.display_name
        : null,
  };
}

export async function upsertMobileProfile(
  profile: MobileProfile,
): Promise<void> {
  const supabase = getMobileSupabaseClient();
  if (!supabase) return;

  const { error } = await supabase.from("profiles").upsert(
    {
      id: profile.id,
      email: profile.email,
      display_name: profile.displayName,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "id" },
  );

  if (error) {
    throw new Error(error.message);
  }
}
