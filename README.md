**Project Structure**
>Data/
>>Country_Data.csv

>Others/
>>Clustering Countries for Strategic Aid Allocation_v2024.1

>>Clustering_Countries_for_Strategic_Aid_Allocation.ipynb

>templates/
>>input_form.html

>>result.html

>app.py

>Clustering Countries for Strategic Aid Allocation.pkl

>requirements.txt

>similar_countries.json(currently for Algeria)


**Setup**
1. Clone The Repository
   ```
   git clone https://github.com/pavannayakanti/scaler_portfolio.git
   cd clustering
   ```
2. Create Virtual Env
    ```
    python3 -m venv .<name>
    source .<name>/scripts/activate
    ```
3. Install Dependencies
   ```
   pip install -r requirements.txt
   ```

**Running the model**

This script will start a Flask API server Opens a UI where we can enter the data and Upon submission shows Countries with Similar Features based on input. Code to run is :  
    ```
    python app.py
    ```

**Blog**
For a detailed analysis and insights from the project, see the Technical Blog at [https://medium.com/@sairams8210/creating-a-clustering-algorithm-for-stratergic-aid-allocation-756cd03dd233.](https://medium.com/@sairams8210/this-blogs-entails-my-personal-experience-on-creating-a-simple-website-that-hosts-a-clustering-756cd03dd233)

**Tableau Dashboard**
For the Tableau Dashboard it can be accessed/viewed at https://public.tableau.com/app/profile/sairam4138/viz/ClusteringCountriesforStrategicAidAllocation/Dasboard
