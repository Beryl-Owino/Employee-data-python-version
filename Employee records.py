#!/usr/bin/env python
# coding: utf-8

# # Employee HR Record
# 
# - In this dataset we are going to analyse a HR record of if employees.
# - Our data is clean and therefore there are no much cleaning to be done for the data,we will do directtly to explotrort analysis and then visualisation of our data.
# 

# In[1]:


#Lets import the labraries we  are going to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import squarify
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from matplotlib import style 
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[2]:


df=pd.read_excel("Employee Final Data.xlsx")
df


# In[3]:


#lets get to know the  size of our data
df.shape


# In[4]:


#Let some basic analysis of our data
df.describe()


# In[5]:


#let check for any duplicates in our data
sum(df.duplicated())


# In[6]:


df["Gender"].info()


# ### Let get gender count of our data

# In[7]:


gender_count=df["Gender"].value_counts()
gender_count 


# In[8]:


#lets plot our result in a donut chart
labels=['female','male']

# colors
colors = ['#FF0000', '#0000FF']

plt.style.use('ggplot')

# Adding Title of chart
plt.title('Employee gender distribution')

# Pie Chart
plt.pie(x=gender_count,labels=labels,colors=colors,
        autopct='%1.1f%%',startangle=90)
plt.axis('equal')

# Add Legends
plt.legend(labels, loc="upper right", title="Gender")

#draw circle
centre_circle = plt.Circle(xy=(0, 0), radius=.70, facecolor='white')

# Adding Circle in Pie chart
plt.gca().add_artist(centre_circle)




# Displaying Chart
plt.show()


# ### Let calculate sum of employee salarie group their enthnicty

# In[ ]:





# In[9]:


#let get the sum of the salaries per each enthnicty
enthnicty_salary=df.groupby('Ethnicity')['Annual_salary'].sum()
enthnicty_salary


# In[10]:


#let represent our data in a pie chart
plt.pie(x=enthnicty_salary, labels = enthnicty_salary.index, startangle = 90,
        counterclock = False, autopct='%1.1f%%');
plt.title('Total salary per Ethnicity')
plt.axis('equal')
plt.show()


# In[ ]:





# ### Calcutate the number of  employees of each position

# In[11]:


#lets find the of employees for each position
position_count=df.groupby('Job_title')['Employee_ID'].count()
position_count.sort_values( ascending=False)


# In[12]:


#let plot a bar graph to present our data 
position_count.sort_values( ascending=False).plot(kind='bar',x='Job_title',y='count',
                                                  title='count of employee per position',color='black',
                                                 figsize=(15,10))


# ### Calculate the sumof wages spent per year in each department and group it per gender

# In[13]:


#let group department,gender and annual salary then find the sum of
#annuall salary per department.

Departmental_wages=df.groupby(['Department','Gender'])['Annual_salary'].sum().reset_index()
Departmental_wages


# In[14]:


# Getting unique departments and genders
departments = Departmental_wages['Department'].unique()
genders = Departmental_wages['Gender'].unique()

# Setting the width of each bar
bar_width = 0.35


# Generating x-axis positions for each department
x = np.arange(len(departments))

# Creating the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
for i, gender in enumerate(genders):
    grouped_Annual_salary = Departmental_wages[Departmental_wages['Gender'] == gender]['Annual_salary']
    ax.bar(x + i * bar_width, grouped_Annual_salary, bar_width, label=gender,)

# Setting the labels for x-axis and y-axis
ax.set_xlabel('Department')
ax.set_ylabel('Annual Salary')

# Setting the x-ticks and labels
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(departments)

# Adding a legend
ax.legend()

# Displaying the bar chart
plt.show()


# In[15]:


#Alternative code(1) by use of pivot tables
#Departmental_wages.pivot_table(index='Department',columns='Gender').plot(kind='bar',y='Annual_salary',figsize=(10, 6))



# In[16]:


#Sammary of gender distribution per department
pd.crosstab(df.Department,df.Gender)


# In[17]:


#let present our result in a combined bar chart

colors = ['blue', 'pink'] 

pd.crosstab(df.Department,df.Gender).plot(kind='bar',figsize=(10,6),color=colors)
plt.title('Gender distribution per department')


# ### Sum of employee salary per country

# In[18]:


#let group our data
country_salary = df.groupby('Country').agg({'Employee_ID': 'count', 'Annual_salary': 'sum'})

# Rename the columns for clarity
country_salary=country_salary.rename(columns={'Employee_ID': 'Employee_count', 'Annual_salary': 'Total_salary'})

country_salary


# In[19]:


# Create a heatmap
sb.heatmap(country_salary, annot=True, fmt="d", cmap='RdYlBu')
plt.title('Employee Count and Total Salary by Country')
plt.xlabel('Metrics')
plt.ylabel('Country')

plt.show()


# ### Tree map representation of the above code

# In[20]:


# Sort the DataFrame by the employee count in descending order
country_salary = country_salary.sort_values('Employee_count', ascending=False)

# Create the treemap
plt.figure(figsize=(10, 8))
squarify.plot(sizes=country_salary['Employee_count'], label=country_salary.index, alpha=0.8)
plt.axis('off')
plt.title('Treemap of Employee Count by Country')

plt.show()


# In[21]:


from astropy.visualization import astropy_mpl_style
astropy_mpl_style['axes.grid'] = False
plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

# Calculate the percentage of total salary for each country
country_salary['salary_percentage'] = country_salary['Total_salary'] / country_salary['Total_salary'].sum()

# Sort the DataFrame by the employee count in descending order
country_salary = country_salary.sort_values('Employee_count', ascending=False)

# Create the treemap
plt.figure(figsize=(10, 8))
squarify.plot(sizes=country_salary['Employee_count'], label=country_salary.index, alpha=0.8)
plt.axis('off')
plt.title('Treemap of Employee Count by Country')

# Disable the grid
plt.grid(False)

# Optionally, you can add a colorbar to represent the percentage of total salary
cmap = plt.cm.get_cmap('YlGnBu')  # Colormap for the colorbar
norm = plt.Normalize(vmin=country_salary['salary_percentage'].min(), vmax=country_salary['salary_percentage'].max())  # Normalize values for colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, format='%.2f', label='Salary Percentage')

plt.show()


# In[22]:


df.head()


# In[23]:


busines_unit=pd.pivot_table(df,index=['Business_unit','Department'],values='Employee_ID',aggfunc='count')
busines_unit


# In[24]:


df.pivot_table(index=['Business_unit','Department'],values='Employee_ID',aggfunc='count').plot(kind='bar',y='Employee_ID',figsize=(10, 6))


# In[25]:


# Create the pivoted table
pivot_table = pd.pivot_table(df, index=['Country', 'City'], values='Employee_ID', aggfunc='count')

# Display the pivoted table
pivot_table

# Add borders to the pivoted table
#styled_table = pivot_table.style.set_properties(**{'border': '1px solid black'})

# Display the styled table
#styled_table


# In[26]:


# Create the pivoted table
pivot_table = pd.pivot_table(df, index=['Country', 'City'], values='Employee_ID', aggfunc='count')

# Calculate the sum of Annual_salary per city
pivot_table['Sum of Annual_salary'] = df.groupby(['Country', 'City'])['Annual_salary'].sum()

# Display the pivoted table
print(pivot_table)


# In[27]:


# Assuming you have a pivot table named 'pivot_table' with 'Country' and 'City' as the index

# Group the pivot table by 'Country' to calculate the total number of employees per country
country_stats = pivot_table.groupby('Country').agg({'Employee_ID': 'sum'})

# Group the pivot table by 'City' to calculate the sum of annual salary per city
city_stats = pivot_table.groupby('City').agg({'Sum of Annual_salary': 'sum'})

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar chart for number of employees per country
ax.bar(country_stats.index, country_stats['Employee_ID'], label='Number of Employees', color='skyblue')
ax.set_ylabel('Number of Employees')

# Set the x-axis ticks
ax.set_xticks(range(len(country_stats)))
ax.set_xticklabels(country_stats.index, rotation=90)

# Create a secondary y-axis for the line graph
ax2 = ax.twinx()

# Create a line graph for sum of annual salary per city
ax2.plot(city_stats.index, city_stats['Sum of Annual_salary'], marker='o', color='red', label='Sum of Annual Salary')
ax2.set_ylabel('Sum of Annual Salary')

# Add a title
plt.title('Number of Employees and Sum of Annual Salary by Country and City')

# Create a single legend for both axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, loc='upper right')

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Display the chart
plt.show()


# ### Let count the sum of bonus per Department

# In[42]:


#let use gruo method to group our department and bonus
Departmental_bonus=df.pivot_table(index='Department',values='Bonus',aggfunc='sum').plot(kind='barh',y='Bonus')


# In[37]:


#Alternative code
#result = df.groupby('Department')['Bonus'].sum().reset_index()
#result 

