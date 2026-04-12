(function navbarComponent() {
  function setActiveLink(container) {
    const currentPath = window.location.pathname;
    const links = container.querySelectorAll('[data-nav-path]');
    links.forEach((link) => {
      const navPath = link.getAttribute('data-nav-path');
      const isActive = currentPath === navPath || (navPath !== '/' && currentPath.startsWith(navPath + '/'));
      link.classList.toggle('active', Boolean(isActive));
    });
  }

  function updateLoginState(container) {
    const isAuthPage = window.location.pathname === '/login' || window.location.pathname === '/register';
    const logoutBtn = container.querySelector('#logoutBtn');
    const darkModeToggle = container.querySelector('#darkmode-toggle');
    const darkModeLabel = container.querySelector('.darkmode-label');

    if (isAuthPage) {
      if (logoutBtn) logoutBtn.style.display = 'none';
      if (darkModeToggle) darkModeToggle.style.display = 'none';
      if (darkModeLabel) darkModeLabel.style.display = 'none';
    }
  }

  async function loadNavbar() {
    const root = document.querySelector('[data-navbar-root]');
    if (!root) return;

    try {
      const response = await fetch('/static/components/navbar.html');
      if (!response.ok) throw new Error('Navbar request failed');
      root.innerHTML = await response.text();
      setActiveLink(root);
      updateLoginState(root);

      if (typeof window.initSharedUi === 'function') {
        window.initSharedUi();
      }
      document.dispatchEvent(new CustomEvent('navbar:loaded'));
    } catch (error) {
      console.error('Failed to load navbar component:', error);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadNavbar);
  } else {
    loadNavbar();
  }
})();