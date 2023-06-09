
# Car-Price-Estimator

This is an end to end Machine Learning project that aims to predict the rough estimate for a used car.

It takes inputs such as Company name, Model,Year of manufacture, Kilometres driven and the fuel type and predicts the price


## Project link
https://p-car-price-estimator.onrender.com/
##  Dataset
This dataset is available here :
 [Dataset](https://github.com/rajtilakls2510/car_price_predictor/blob/master/quikr_car.csv)

## Tech Stack

- Python
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Sci-kit learn
- Pickle
- Flask
- HTML
- Gunicorn
- Waitress



## Installation

Install all the necessary libraries using pip

```bash
  pip install -r requirements.txt
```
    
##  Working
1)Imported all the necessary libraries and the dataset.

2)Checked the distribution and flow of the data.

3)Performed intensive EDA for easier interpretation of the data.

4)Visualized the data.

5)Performed Data Wrangling.

6)Modelling with baseline regression model.

7)Checking the performance with other regression models.

8)Going with the model that best explains the variance .

9)Dumping the model in a pickle file .

10)Deploying it in flask. 

11)Deploying the application on render platform.
## Deployment

To deploy this project run

```bash
  gunicorn app:app
```


## Author

- [@Pavan Kumar V](https://github.com/Pavan-477)

