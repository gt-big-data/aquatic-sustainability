const registerForm = document.getElementById("registerForm");
if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
        console.log("Attempting register");
        e.preventDefault();
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;
        const confirmPassword = document.getElementById("confirmPassword").value;

        if (password !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }

        try {
            const res = await fetch("/api/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            let data = null;
            try { data = await res.json(); } catch (err) { /* ignore json parse errors */ }

            if (res.ok) {
                alert("Registration successful! Please check your email if verification is required.");
                window.location.href = "/login";
            } else {
                const msg = (data && (data.error || data.message)) || `Registration failed (${res.status})`;
                alert(msg);
            }
        } catch (err) {
            console.error("Registration error", err);
            alert("Network error while attempting to register. Check your connection and try again.");
        }
    });
}