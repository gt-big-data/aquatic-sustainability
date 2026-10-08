/**
 * auth-google.js
 * Google OAuth via Supabase — works on both login.html and register.html.
 * Reads Supabase credentials dynamically from the /api/config/supabase endpoint
 * so we never expose secrets in static JS files.
 */

(async () => {
    // ── 1. Load Supabase credentials from backend ────────────────────────────
    let supabaseUrl = "";
    let supabaseAnonKey = "";

    try {
        const res = await fetch("/api/config/supabase");
        if (res.ok) {
            const cfg = await res.json();
            supabaseUrl = cfg.supabaseUrl || "";
            supabaseAnonKey = cfg.supabaseAnonKey || "";
        }
    } catch (err) {
        console.warn("Could not load Supabase config:", err);
    }

    if (!supabaseUrl || !supabaseAnonKey) {
        console.warn("Supabase not configured — Google OAuth button disabled.");
        const btns = document.querySelectorAll(".btn-google");
        btns.forEach(btn => {
            btn.disabled = true;
            btn.title = "Authentication service not configured";
        });
        return;
    }

    // ── 2. Initialise Supabase client ────────────────────────────────────────
    const { createClient } = window.supabase;
    const sb = createClient(supabaseUrl, supabaseAnonKey);

    // ── 3. Determine redirect URL ─────────────────────────────────────────────
    // After Google redirects back to Supabase and Supabase redirects to us,
    // we want to land on /auth/callback which then forwards the user home.
    const redirectTo = `${window.location.origin}/auth/callback`;

    // ── 4. Wire up buttons ────────────────────────────────────────────────────
    const googleBtns = document.querySelectorAll("#googleLoginBtn, #googleRegisterBtn");

    googleBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            btn.disabled = true;
            btn.style.opacity = "0.7";

            const { error } = await sb.auth.signInWithOAuth({
                provider: "google",
                options: { redirectTo },
            });

            if (error) {
                console.error("Google OAuth error:", error.message);
                btn.disabled = false;
                btn.style.opacity = "1";
                alert("Google sign-in failed: " + error.message);
            }
            // On success Supabase redirects the browser automatically — no further action needed.
        });
    });
})();
