const axios = require('axios');

const apiUrl = 'http://localhost:5000/predict';

// Input data for prediction
const data = {
  area: 'Super built-up Area',
  location: 'Electronic City Phase II',
  size: 2,
  sqft: 1056,
  bath: 2,
  balcony: 2,
};
const queryString = Object.entries(data)
  .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
  .join('&');

axios
  .post(apiUrl, queryString)
  .then((response) => {
    console.log('Prediction:', response.data);
  })
  .catch((error) => {
    console.error('Error:', error.message);
  });
