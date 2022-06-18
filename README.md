# Nawy-Qualified-leads


# Our Client 
Nawy is a one-stop-shop that will provide you with the key to your new home by presenting all the necessary information and helps you make an informed decision via reliable brokers. Users are able to search for available units by selecting criteria that match their needs.  We are partnered with all the big real estate developers. Our team includes experienced real estate brokers that facilitate the transaction. 

There is a lack of information in the Egyptian real estate market. Making an informed decision requires time-consuming manual work and comparing between available options is a difficult task. This creates an unnecessary time burden and leads to incorrect decision making when buying/selling properties. Brokers in Egypt have a bad reputation and are perceived to be untrustworthy.


# Defining the Problem and Project Goal
In Nawy we are initiating targeting campaigns on a daily basis to generate Real estate leads.
The main pain is generating qualified leads.

   * The first objective of this project is to recognize key factors that will use to know qualified lead.
   * So we need to build a model to classify the received leads into two categories: low qualified and high qualified leads.


------------------------------------------------------------------------------------------------------------------
# During this project, I went through these steps below :

* Objectives
* Data Understanding
* Data Pre-processing
* Challenges
* Data Exploration & Hypothesis testing
* Data Preparation
  * - Handling non-realistics data, outliers and   missing values.
  * - Feature Selection and Scaling
  * - Feature Engineering 
  * - Handling imbalance data (undersampling or oversamplimg)
* Model Selection & Evaluation metric
  * - Fine-Tune the model
  * - Hyperparameters Optimization
  * - Feature Importance selections
* Results
* Other Approaches
* Deployment using flask
* Business Recommendations
#### - [Presentation Link]([https://drive.google.com/file/d/1mWW14-DHi0TUWcxMHVEowokwnn2Z9jvZ/view?usp=sharing](https://drive.google.com/file/d/1gVvJrAe0MN0j1y16uuMcTaFI74Mgssqy/view?usp=sharing)
------------------------------------------------------------------------------------------------------------------

# Data Dictionary
* **lead_id** Is unique id representing customer (customer my reach more than one time)
* **customer_name** Name of the customer
* **lead_mobile_network** To differentiate between local and international customers
* **message** Message left by the customer
* **lead_time** Time of receiving the lead
* **method_of_contact** Contact method used to reach us
* **ad_group** Include information of target audience used in the campaign
* **lead_source** Channel used to reach us
* **campaign** Name of the targeting campaign
* **Location** The location which the customer is searching for
* **Low_qualified** Define if the lead is low or high qualified


## Instructions :-
1) Install "requirement.txt" Packages :- 
   - "pip install -r requirements.txt"
2) Download " Dataset".
   - [Dataset](https://drive.google.com/file/d/1fVKBF2QPBTAmlf_FOgyqRBku43a9j26f/view?usp=sharing)
3) Place "dataset.csv" and "Behaviour.tsv" in "./dataset/".
4) Run "app.py".
5) Open http://127.0.0.1:5000/
