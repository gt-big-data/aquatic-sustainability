// News Page JavaScript
// This file handles news-specific functionality

document.addEventListener('DOMContentLoaded', function() {
  // News page specific logic can be added here
  
  // Handle logout
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', function() {
      // Handle logout logic
      window.location.href = '/login';
    });
  }
});
