// Dark mode toggle with localStorage persistence
(function () {
  var toggle = document.getElementById('theme-toggle');
  var root = document.documentElement;
  var stored = localStorage.getItem('theme');
  if (stored) {
    root.setAttribute('data-theme', stored);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  }
  if (toggle) {
    toggle.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }

  // Mobile sidebar toggle
  var sidebarToggle = document.getElementById('sidebar-toggle');
  var sidebar = document.getElementById('sidebar');
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });
    // Close sidebar when clicking a link (mobile)
    sidebar.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        sidebar.classList.remove('open');
      }
    });
  }
})();
