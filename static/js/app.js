document.addEventListener('DOMContentLoaded', function () {
    // Example: Change background color when button is clicked
    const button = document.getElementById('change-color-button');
    button.addEventListener('click', function () {
        document.body.style.backgroundColor = 'lightblue';
    });

    // Example: Display alert when a form is submitted
    const form = document.getElementById('personality-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        alert('Form Submitted!');
    });
});



document.getElementById("personality-form").addEventListener("click", function(e) {
    const inputs = document.querySelectorAll(".question input");
    let allFilled = true;

    inputs.forEach(input => {
        if (input.value < 1 || input.value > 5 || input.value === "") {
            allFilled = false;
        }
    });

    if (!allFilled) {
        e.preventDefault();
        alert("Please ensure all questions are answered correctly (1-5).");
    }
});