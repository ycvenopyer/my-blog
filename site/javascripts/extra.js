// 首页标题：鼠标光晕跟随
function initHeroTitle() {
    const hero = document.querySelector('.hero-title');
    if (!hero || hero.dataset.heroInit) return;
    hero.dataset.heroInit = '1';

    const glow = hero.querySelector('.hero-title__glow');
    if (!glow) return;

    hero.addEventListener('mousemove', function(e) {
        const rect = hero.getBoundingClientRect();
        glow.style.left = (e.clientX - rect.left) + 'px';
        glow.style.top = (e.clientY - rect.top) + 'px';
        hero.classList.add('hero-title--active');
    });

    hero.addEventListener('mouseleave', function() {
        hero.classList.remove('hero-title--active');
    });
}

if (typeof document$ !== 'undefined') {
    document$.subscribe(initHeroTitle);
} else {
    document.addEventListener('DOMContentLoaded', initHeroTitle);
}

// 回到顶部按钮
document.addEventListener('DOMContentLoaded', function() {
    // 创建回到顶部按钮
    const backToTop = document.createElement('button');
    backToTop.className = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.style.display = 'none';
    backToTop.title = '回到顶部';
    document.body.appendChild(backToTop);

    // 滚动时显示/隐藏按钮
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTop.style.display = 'flex';
        } else {
            backToTop.style.display = 'none';
        }
    });

    // 点击回到顶部
    backToTop.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
});

// 搜索增强
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('input[type="search"]');
    if (searchInput) {
        searchInput.placeholder = '搜索文章...';
    }
});

// 外部链接在新窗口打开
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="http"]');
    links.forEach(link => {
        if (!link.getAttribute('href').startsWith(window.location.origin)) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

// 图片懒加载支持
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.loading = 'lazy';
    });
});
