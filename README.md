# Deloitte Machine Learning Challenge 2021
Deloitte Presents Machine Learning Challenge: Predict Loan Defaulters in association with [Machine Hack](https://machinehack.com/hackathon/deloitte_presents_machine_learning_challenge_predict_loan_defaulters/overview)
## Overview
* Banks run into losses when a customer doesn't pay their loans on time. Because of this, every year, banks have losses in crores, and this also impacts the country's economic growth to a large extent. In this hackathon, we look at various attributes such as funded amount, location, loan, balance, etc., to predict if a person will be a loan defaulter or not. 

* To solve this problem, MachineHack has created a training dataset of 67,463 rows and 35 columns and a testing dataset of 28,913 rows and 34 columns. The hackathon demands a few pre-requisite skills like big dataset, underfitting vs overfitting, and the ability to optimise “log_loss” to generalise well on unseen data. 
## Data Description

| Feature             
| ----------------------- 
| Customer_ID             
 ID                            
 Loan Amount                  
 Funded Amount                 
 Funded Amount Investor       
 Term                          
 Batch Enrolled               
 Interest Rate               
 Grade                         
 Sub Grade                    
 Employment Duration          
 Home Ownership                
 Verification Status           
 Payment Plan                
 Loan Title                   
 Debit to Income              
 Delinquency - two years      
 Inquires - six months         
 Open Account                  
 Public Record                 
 Revolving Balance             
 Revolving Utilities         
 Total Accounts                
 Initial List Status           
 Total Received Interest       
 Total Received Late Fee       
 Recoveries                    
 Collection Recovery Fee       
 Collection 12 months Medical  
 Application Type             
 Last week Pay                
 Accounts Delinquent             
 Total Collection Amount       
 Total Current Balance         
 Total Revolving Credit Limit  
 Loan Status      
 
 ## Skills Practiced
 * Visualization
 * Feature Engineering- different dncoding techniques
 * Multiple models-hyperparameter tuning
 * Methods to maximize the metric of evaluation
 * Stacking 
 
 
 
 import pandas as pd
from pptx import Presentation
from pptx.util import Inches

# Create a sample dataframe
df = pd.DataFrame({
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'Age': [30, 25, 35],
    'Occupation': ['Engineer', 'Teacher', 'Doctor']
})

# Create a PowerPoint presentation and add a slide
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[1])

# Set the location, height, and width of the table
left = Inches(1)
top = Inches(2)
height = Inches(4)
width = Inches(6)

# Add a table to the slide
table = slide.shapes.add_table(
    rows=df.shape[0]+1, cols=df.shape[1],
    left=left, top=top, width=width, height=height
)

# Set the column widths based on the maximum text width for each column
max_col_widths = [0] * df.shape[1]
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        cell = table.cell(i+1, j)
        text_width = cell.text_frame.paragraphs[0].runs[0].font.size * len(cell.text)
        max_col_widths[j] = max(max_col_widths[j], text_width)
for idx, width in enumerate(max_col_widths):
    table.columns[idx].width = width

# Write the column names to the table
for i, col_name in enumerate(df.columns):
    cell = table.cell(0, i)
    cell.text = col_name
    cell.text_frame.paragraphs[0].font.size = Inches(0.15)

# Write the data to the table
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        cell = table.cell(i+1, j)
        cell.text = str(df.iloc[i, j])
        cell.text_frame.paragraphs[0].font.size = Inches(0.15)

# Save the presentation
prs.save('table.pptx')

