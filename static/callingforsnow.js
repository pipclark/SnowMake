/*const snowbutton = document.getElementById("main-btn") /* targets the mobile menu tag */
//const loadingtext = document.getElementById("hiddenmessage")
const snowbutton = document.querySelector(".main__btn")
//const snowbuttontext = document.querySelector(".main__btn a")
const snowflakegif = document.getElementById("main__img")
const snowbuttontext = document.getElementById("main-btn-text")


snowbutton.addEventListener('click', function() {
    console.log('snow generating button pressed')
    snowbutton.classList.toggle('is-active'); /*. is-active seems to hold it and just active is just while clicking ? toggles between active and normal css */
    snowbutton.disabled = true;
    snowflakegif.src = "/static/images/growing.gif"
    $.ajax({        
        url: '/randomflake',
        //url: 'http://127.0.0.1:5000/randomflake',
        type: 'GET',
        contentType: "image/gif",
        success: function(result) {
            console.log('called server to generate and send snow flake')
            snowflakegif.src = 'data:image/gif;base64,' + result;
            snowbutton.disabled = false;
        }
    });     
    snowbuttontext.innerHTML = 'Make me another snowflake!'

});





    
    

