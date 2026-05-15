import { afterEach, describe, expect, it, vi } from "vitest";

describe("Supabase admin client", () => {
  const originalUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const originalServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  afterEach(() => {
    if (originalUrl === undefined) {
      delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    } else {
      process.env.NEXT_PUBLIC_SUPABASE_URL = originalUrl;
    }
    if (originalServiceRoleKey === undefined) {
      delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    } else {
      process.env.SUPABASE_SERVICE_ROLE_KEY = originalServiceRoleKey;
    }
    vi.resetModules();
    vi.doUnmock("@supabase/supabase-js");
    vi.restoreAllMocks();
  });

  it("requires the public Supabase URL and service role key", async () => {
    delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    vi.doMock("@supabase/supabase-js", () => ({ createClient: vi.fn() }));

    const { getSupabaseAdminClient } =
      await import("@/lib/server/supabaseAdmin");

    expect(() => getSupabaseAdminClient()).toThrow(
      "Missing required Supabase environment variable: NEXT_PUBLIC_SUPABASE_URL",
    );
  });

  it("creates one non-persistent server client and reuses it", async () => {
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    const client = { from: vi.fn() };
    const createClient = vi.fn().mockReturnValue(client);
    vi.doMock("@supabase/supabase-js", () => ({ createClient }));

    const { getSupabaseAdminClient } =
      await import("@/lib/server/supabaseAdmin");

    expect(getSupabaseAdminClient()).toBe(client);
    expect(getSupabaseAdminClient()).toBe(client);
    expect(createClient).toHaveBeenCalledTimes(1);
    expect(createClient).toHaveBeenCalledWith(
      "https://example.supabase.co",
      "service-role-key",
      {
        auth: {
          persistSession: false,
          autoRefreshToken: false,
        },
      },
    );
  });
});
