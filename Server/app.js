const axios = require('axios');

// Replace 'http://localhost:5000/predict' with your Flask API endpoint
const apiUrl = 'http://localhost:5000/predict';

// Input data for prediction
const data = {
  area_type: 'Super built-up  Area',
  location: 'Electronic City Phase II',
  size: '2 BHK',
  total_sqft: 1056,
  bath: 2,
};

axios
  .post(apiUrl, data)
  .then((response) => {
    console.log('Prediction:', response);
  })
  .catch((error) => {
    console.error('Error:', error.message);
  });
