const progressBar = document.getElementById('reading-progress-bar');
const tocLinks = [...document.querySelectorAll('.toc a')];
const sections = tocLinks.map((link) => document.querySelector(link.getAttribute('href'))).filter(Boolean);
const themeToggle = document.querySelector('.theme-toggle');
const syncProgress = () => {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
  if (progressBar) progressBar.style.width = `${Math.min(100, Math.max(0, ratio))}%`;
};
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    const id = `#${entry.target.id}`;
    const link = tocLinks.find((item) => item.getAttribute('href') === id);
    if (!link) return;
    if (entry.isIntersecting) {
      tocLinks.forEach((item) => item.classList.remove('active'));
      link.classList.add('active');
    }
  });
}, { rootMargin: '-35% 0px -55% 0px', threshold: 0.1 });
sections.forEach((section) => observer.observe(section));
window.addEventListener('scroll', syncProgress, { passive: true });
window.addEventListener('load', syncProgress);
const savedTheme = localStorage.getItem('portfolio-os-theme');
if (savedTheme) document.body.dataset.theme = savedTheme;
if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = next;
    localStorage.setItem('portfolio-os-theme', next);
  });
}
