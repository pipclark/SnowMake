

const snowbutton = document.getElementById("main-btn")

const snowflakegif = document.getElementById("main__img")
const snowbuttontext = document.getElementById("main-btn-text")

const graphbutton = document.getElementById("graph-btn")
const graphgif = document.getElementById("second__img")
const graphbuttontext = document.getElementById("graph-btn-text")

snowbutton.addEventListener('click', function() {
    console.log('snow generating button pressed')
    snowbutton.classList.toggle('is-active'); /*. is-active seems to hold it and just active is just while clicking ? toggles between active and normal css */
    snowbutton.disabled = true;
    snowflakegif.src = "/static/images/growing.gif"

    var inputdata = { // set up dictionary for sending data
        height: null,
        humidity: null,
        branchopt: 0,
        sizeopt: 0,
    }
    inputdata.height = document.getElementById('height').value;
    inputdata.humidity = document.getElementById('humidity').value;
    if(document.getElementById('branchswitch').checked){
        inputdata.branchopt = 1;
    }
    if(document.getElementById('sizeswitch').checked){
        inputdata.sizeopt = 1;
    }

    console.log(inputdata);

    $.ajax({
        url: '/randomflake',
        //url: 'http://127.0.0.1:5000/randomflake',
        type: 'POST',
        data: JSON.stringify(inputdata),
        contentType: "image/gif",
        success: function(result) {
            console.log('posted inputs to server to generate and send snow flake')
            snowflakegif.src = 'data:image/gif;base64,' + result;
            snowbutton.disabled = false;
        }
    })
   
    snowbuttontext.innerHTML = 'Make me another snowflake!'

});


graphbutton.addEventListener('click', function() {
    console.log('graph generating button pressed')
    graphbutton.classList.toggle('is-active'); /*. is-active seems to hold it and just active is just while clicking ? toggles between active and normal css */
    graphgif.src = "/static/images/graphforming.gif"
    graphbutton.disabled = true;

    var inputdata = { // set up dictionary for sending data
        height: null,
        humidity: null,
        branchopt: 0,
    }
    inputdata.height = document.getElementById('height').value;
    inputdata.humidity = document.getElementById('humidity').value;
    if(document.getElementById('branchswitch').checked){
        inputdata.branchopt = 1;
    }
    
    //const inputs = document.querySelectorAll('.controls input').values;
    //inputs.forEach(input => input.value(sendvalues));
    console.log(inputdata);

    $.ajax({
        url: '/advflakegraphs',
        //url: 'http://127.0.0.1:5000/advflakegraphs',
        type: 'POST',
        data: JSON.stringify(inputdata),
        contentType: "image/gif",
        success: function(result) {
            console.log('posted inputs to server to generate and send snow flake')
            graphgif.src = 'data:image/gif;base64,' + result;
            graphbutton.disabled = false;
        }
    })
   
    graphbuttontext.innerHTML = 'Make me more graphs! (change conditions first)'

});