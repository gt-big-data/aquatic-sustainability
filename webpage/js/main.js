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
});