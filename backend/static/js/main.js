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

	// Logout button logic (optional auth)
	const logoutBtn = document.getElementById('logoutBtn');
	const loginLink = document.getElementById('loginLink');
	const token = localStorage.getItem('access_token');

	if (logoutBtn) {
		logoutBtn.addEventListener('click', () => {
			localStorage.removeItem('access_token');
			// After logout, send them to login
			window.location.href = '/login';
		});
	}

	// Toggle auth UI but do NOT force redirect if no token
	if (token) {
		if (logoutBtn) logoutBtn.style.display = 'inline-flex';
		if (loginLink) loginLink.style.display = 'inline-flex'; // will hide below for login/register
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