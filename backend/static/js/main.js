// Year in footer
document.addEventListener('DOMContentLoaded', () => {
	const yearEl = document.getElementById('year');
	if (yearEl) yearEl.textContent = new Date().getFullYear();

	// Theme toggle (persist in localStorage)
	const root = document.documentElement;
	const themeToggle = document.getElementById('themeToggle');
	const saved = localStorage.getItem('theme');
	if (saved === 'light') root.classList.add('light');
	if (themeToggle) {
		themeToggle.addEventListener('click', () => {
			root.classList.toggle('light');
			localStorage.setItem('theme', root.classList.contains('light') ? 'light' : 'dark');
		});
	}

	// Logout + auth UI logic (optional auth)
	const loginLink = document.getElementById('loginLink');
	const logoutBtn = document.getElementById('logoutBtn');
	const token = localStorage.getItem('access_token');

	// When clicking Logout: clear token, then go to /login
	if (logoutBtn) {
		logoutBtn.addEventListener('click', (e) => {
			e.preventDefault(); // prevent default link behavior
			localStorage.removeItem('access_token');
			window.location.href = '/login';
		});
	}

	// Show only Logout when logged in; only Login when logged out
	if (token) {
		if (logoutBtn) logoutBtn.style.display = 'inline-flex';
		if (loginLink) loginLink.style.display = 'none';
	} else {
		if (logoutBtn) logoutBtn.style.display = 'none';
		if (loginLink) loginLink.style.display = 'inline-flex';
	}

	// If already logged in and on login/register, redirect to home
	const isLogin = window.location.pathname.includes('/login');
	const isRegister = window.location.pathname.includes('/register');
	if (token && (isLogin || isRegister)) {
		window.location.href = '/';
	}

});
