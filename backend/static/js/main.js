function initSharedUi() {
	const yearEl = document.getElementById('year');
	if (yearEl) yearEl.textContent = new Date().getFullYear();

	// Theme toggle (persist in localStorage)
	const root = document.documentElement;
	const themeToggle = document.getElementById('darkmode-toggle');
	const saved = localStorage.getItem('theme');
	if (themeToggle) {
		if (saved === 'light') {
			root.classList.add('light');
			themeToggle.checked = false;
		} else {
			themeToggle.checked = true;
		}

		if (themeToggle.dataset.bound !== 'true') {
			themeToggle.addEventListener('change', () => {
				if (themeToggle.checked) {
					root.classList.remove('light'); // dark mode
					localStorage.setItem('theme', 'dark');
				} else {
					root.classList.add('light'); // light mode
					localStorage.setItem('theme', 'light');
				}
			});
			themeToggle.dataset.bound = 'true';
		}
	}

	// Logout button logic (add to all protected pages)
	const logoutBtn = document.getElementById('logoutBtn');
	if (logoutBtn && logoutBtn.dataset.bound !== 'true') {
		logoutBtn.addEventListener('click', () => {
			localStorage.removeItem('access_token');
			window.location.href = '/login';
		});
		logoutBtn.dataset.bound = 'true';
	}

}

// Year in footer
document.addEventListener('DOMContentLoaded', () => {
	initSharedUi();

	// Protect all pages except login/register
	const isLogin = window.location.pathname.includes('/login');
	const isRegister = window.location.pathname.includes('/register');
	const token = localStorage.getItem('access_token');
	if (!isLogin && !isRegister) {
		// Protected page: redirect to login if not authenticated
		//TODO: Delete this local storage thing for skipping login (just for dev convenience) - Josh
		if (!localStorage.getItem("skippedLogin") && !token) {
			window.location.href = '/login';
		}
	} else {
		// If on login or register and already authenticated, redirect to index
		if (token) {
			window.location.href = '/';
		}
	}
});

window.initSharedUi = initSharedUi;