// Scroll-spy: highlight the TOC entry for the section currently in view.
const sections = Array.from(document.querySelectorAll('main section'));
const tocLinks = new Map(
  Array.from(document.querySelectorAll('aside.toc a')).map((a) => [a.getAttribute('href').slice(1), a])
);

function setActive(id) {
  tocLinks.forEach((link) => link.classList.remove('active'));
  const active = tocLinks.get(id);
  if (active) active.classList.add('active');
}

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((e) => e.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
    if (visible.length > 0) setActive(visible[0].target.id);
  },
  { rootMargin: '-20% 0px -60% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] }
);

sections.forEach((s) => observer.observe(s));

// Default to first section on load
if (sections.length > 0) setActive(sections[0].id);
