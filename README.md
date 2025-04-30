Files:

```api_data_fetch.py``` - script used to fetch data from the OpenWeatherMap APIs.

```feature_repo/feature_definitions.py``` - This script is responsible for creating the feature store.

```feature_repo/aqi_workflow.py``` - This script stores the data into the feast feature store.

```model_training.py``` - This script trains and evaluates 3 different models.

```feature_repo/aqi_predictions.py``` - This script takes the best performing model from the ``model_training.py`` script and uses that to predict future data. This script also has the SHAP explanations.

```streamlit_frontend.py``` - This script creates the front-end interface.

```frontend_interface.py``` - This script launches the FASTapi and passes the data from the aqi_prediction.csv to the API that is then displayed through the Streamlit interface.

To run the project, you're first required to run the ```api_data_fetch.py```. After that is done, you are to locate your terminal to the ```feature_repo``` folder and then run the command ```feast apply```. After that is done, you are to run the ```aqi_workflow.py``` script. After that is done, locate back to the main folder and then run the ```model_training.py``` script. And after that, go back to the ```feature_repo``` and run the ```aqi_prediction.py``` script. Now lastly, move back to the main folder and run the ```frontend_interface.py``` this will generate an interface and the project will be fully running.