const loginForm = document.getElementById("loginForm");
if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
        console.log("Attempting login");
        e.preventDefault();
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        try {
            // Use relative path so the same origin works in many deploy setups.
            const res = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            let data = null;
            try { data = await res.json(); } catch (err) { /* non-json response */ }

            if (res.ok) {
                // store token if present
                if (data && data.access_token) localStorage.setItem("access_token", data.access_token);
                // simple UX: redirect to root (adjust as needed)
                window.location.href = "/";
            } else {
                const msg = (data && (data.error || data.message)) || `Login failed (${res.status})`;
                alert(msg);
            }
        } catch (err) {
            console.error("Login error", err);
            alert("Network error while attempting to log in. Check your connection and try again.");
        }
    });
}