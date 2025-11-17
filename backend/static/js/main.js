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

	// Logout button logic (add to all protected pages)
	const logoutBtn = document.getElementById('logoutBtn');
	if (logoutBtn) {
		logoutBtn.addEventListener('click', () => {
			localStorage.removeItem('access_token');
			window.location.href = '/login';
		});
	}

	// Protect all pages except login/register
	const isLogin = window.location.pathname.includes('/login');
	const isRegister = window.location.pathname.includes('/register');
	const token = localStorage.getItem('access_token');
	if (!isLogin && !isRegister) {
		// Protected page: redirect to login if not authenticated
		if (!token) {
			window.location.href = '/login';
		}
	} else {
		// If on login or register and already authenticated, redirect to index
		if (token) {
			window.location.href = '/';
		}
	}
});