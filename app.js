document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    let data = {
        age: document.getElementById('age').value,
        sleep_duration: document.getElementById('sleep_duration').value,
        rem_sleep_percentage: document.getElementById('rem_sleep_percentage').value,
        deep_sleep_percentage: document.getElementById('deep_sleep_percentage').value,
        light_sleep_percentage: document.getElementById('light_sleep_percentage').value,
        awakenings: document.getElementById('awakenings').value,
        caffeine_consumption: document.getElementById('caffeine_consumption').value,
        alcohol_consumption: document.getElementById('alcohol_consumption').value,
        exercise_frequency: document.getElementById('exercise_frequency').value
    };
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('result').innerText = result.prediction === 1 
            ? 'You are likely to experience insomnia.' 
            : 'You are not likely to experience insomnia.';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
