const menu = document.querySelector('#mobile-menu') /* targets the mobile menu tag */
const menuLinks = document.querySelector('.navbar__menu')

menu.addEventListener('click', function() {
    menu.classList.toggle('is-active');
    menuLinks.classList.toggle('active');
});