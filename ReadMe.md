```python
import pandas as pd
from pandas import Series, DataFrame
```


```python
#Import dataset
titanic_df = pd.read_csv('/Users/alexreynolds/Documents/Data Projects/Titanic Project/train.csv')
```


```python
#Preview the Data
titanic_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>

```python
#General info on dataset
titanic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB


# Questions to analyze

1. Who were the passengers on the Titanic? (Age, gender, class, etc)
2. What deck were the passengers on and how does that relate to their class?
3. Where did the passengers come from?
4. Who was alone and who as with family?

# Deeper, broader question:

5. What factors helped someone survive?

## Who were the passengers on the titanic?


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
#Check gender
sns.countplot(x='Sex', data=titanic_df)
```


![png](output_7_1.png)
    

```python
#Seperate gender by classes
sns.countplot(x='Sex',data=titanic_df,hue='Pclass')
```


![png](output_8_1.png)
    


Quite a few more males in the 3rd class than females, an interesting find. However, it might be useful to know the split between males,females,and children.


```python
#Visualize gender in each class
sns.countplot(x='Pclass',data=titanic_df,hue='Sex')
```


![png](output_10_1.png)
    

```python
#Find number of children on board
#Define anyone under 16 as a child

def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex
```


```python
#New column called 'person' to list male, female, or child 
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
```


```python
titanic_df.head(10)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
      <th>Alone</th>
      <th>Survior</th>
      <th>Survivor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>With Family</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
      <td>Alone</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
      <td>child</td>
      <td>With Family</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
      <td>child</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>

```python
#Visualize classes again with childnren included
sns.countplot(x='Pclass',data=titanic_df,hue='person')
```




![png](output_14_1.png)


Higher amount of children seen in 3rd class and not so many in 1st class


```python
#Distribuiton of ages on board
titanic_df['Age'].hist(bins=70)
```




![png](output_16_1.png)
    

```python
#Average age on board
titanic_df['Age'].mean()
```


    29.69911764705882


```python
#Number of male, female, child on board
titanic_df['person'].value_counts()
```


    male      537
    female    271
    child      83
    Name: person, dtype: int64


```python
#Use FacetGrid to visualize gender by age 

fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()
```




![png](output_19_1.png)
    

```python
#Same FacetGrid with children included

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()
```




![png](output_20_1.png)
    

```python
#FacetGrid for class

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()
```




​    
![png](output_21_1.png)
​    


We see the follwing with the passengers on board:
1. More males than females
2. Significanly more males than females in 3rd class
3. Highest amount of children in 3rd class
4. Average age of passenger of 29.6
5. Younger passengers tended to be more lower classes rather than 1st class

## What deck were the passengers on and how does that relate to their class?


```python
#Preview dataset again
titanic_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
    </tr>
  </tbody>
</table>

```python
#Drop the NaN values in cabin column and create new object
deck = titanic_df['Cabin'].dropna()
```


```python
#Preview deck
deck.head()
```


    1      C85
    3     C123
    6      E46
    10      G6
    11    C103
    Name: Cabin, dtype: object


```python
#Get first letter for the deck level
levels =[]

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns=['Cabin']
cabin_df.sort_values(by='Cabin')

#Visualize number of passengers per deck
sns.countplot('Cabin',data=cabin_df,palette="winter_d",order=['A','B','C','D','E','F','G'])
```


​    
![png](output_27_2.png)
​    


Most passengers seen in B and C decks.

## Where did the passengers come from?


```python
#Visualize where passengers come from
sns.countplot('Embarked', data=titanic_df, hue='Pclass', order=['C','Q','S'])
```




​    
![png](output_30_2.png)
​    


Most passengers embarked from S which was from Southhampton, with C (Cherbourg) next highest and finally Q (Queenstown) as the lowest 

Intesting to see that queenstown had slmost all 3rd class passengers. Would be interesting to see the economics of all the cities during that time period to see correlation between the two

## Who was alone and who as with family?


```python
titanic_df.head(10)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
      <td>child</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
      <td>child</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python
#Add new column to deteremine if passenger was alone or not
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
```


```python
titanic_df['Alone']
```


    0      1
    1      1
    2      0
    3      1
    4      0
          ..
    886    0
    887    0
    888    3
    889    0
    890    0
    Name: Alone, Length: 891, dtype: int64


```python
#Classify with family or alone
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
```

```python
#Verify it worked
titanic_df['Alone']
```


    0      With Family
    1      With Family
    2            Alone
    3      With Family
    4            Alone
              ...     
    886          Alone
    887          Alone
    888    With Family
    889          Alone
    890          Alone
    Name: Alone, Length: 891, dtype: object


```python
#Visualize passenger alone vs with family
sns.countplot('Alone', data=titanic_df,palette='Blues')
```


![png](output_39_2.png)


See more passengers alone rather than with family

## What factors helped someone survive?


```python
#Create new column to easily see if passenger survived 
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})
```


```python
titanic_df['Survivor']
```


    0       no
    1      yes
    2      yes
    3      yes
    4       no
          ... 
    886     no
    887    yes
    888     no
    889    yes
    890     no
    Name: Survivor, Length: 891, dtype: object


```python
#Visual of those who survived and those who did not 
sns.countplot('Survivor',data=titanic_df,palette='Set1')
```




​    
![png](output_44_2.png)
​    


Can clearly see that there were more that died than survived. Lets see if the class of the passenger had an effect on the survival rate.


```python
#Visualize those who survived based on class
sns.factorplot('Pclass','Survived',data=titanic_df)
```




​    
![png](output_46_2.png)
​    


Survival rate of those in the 3rd class was significantly less than those in 1st class. There were more men in 3rd class than in first so maybe that was a factor into this lower rate 


```python
#Visualize those who survived based on gender/child
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)
```




​    
![png](output_48_2.png)
​    


From this visualization we can see that being a male or being in 3rd class had much lower survival rates.  Regardless of class, being a male in any class lead to dramatic decrease of survival rate compared to the rest of the passengers.  


```python
#Visualize age vs survival rate
sns.lmplot('Age','Survived',data=titanic_df)
```




​    
![png](output_50_2.png)
​    


See a general trend that the older the passenger was, the less likely they survived.  Lets look at this trend based on the their class and age.


```python
#Linear plot of age vs survival rate with class separation
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df, palette='winter')
```




​    
![png](output_52_2.png)
​    


See that trend stays the same but passengers in lower classes had lower surviving rate as already discovered.


```python
#Clean up last visualization by creating bins
generations = [10,20,40,60,80]
```


```python
#Use bins to clean up visualization
sns.lmplot('Age','Survived',hue = 'Pclass', data=titanic_df, palette='winter',x_bins = generations).set(xlim=(0,90))
```




​    
![png](output_55_2.png)
​    


See a large standard deviation on older passengers in first class. Lets see if gender plays a role in this...


```python
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations).set(xlim=(0,90))
```




​    
![png](output_57_2.png)
​    


We see that the older males were the lower the survival rate with females being the opposite.  The older the female the higher the survival rate.  

We have seen how the higher class passengers has higher survival rates as well has women and children.

## Did the deck have an effect on the passenger survival rate?


```python
#Preview dataset again
titanic_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
      <th>Alone</th>
      <th>Survior</th>
      <th>Survivor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>With Family</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>female</td>
      <td>Alone</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
    </tr>
  </tbody>
</table>



```python
#Drop NaN in cabin column and create new dataframe with this info
cabin_df = titanic_df.dropna(subset=['Cabin'])
```


```python
#Preview new dataframe
cabin_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
      <th>Alone</th>
      <th>Survior</th>
      <th>Survivor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
      <td>child</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
      <td>female</td>
      <td>Alone</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>

```python
#Make new object with cabin info
deck = cabin_df['Cabin']
```


```python
#Add new column 'Deck' with letter of deck passenger stayed at
levels = []
for level in deck:
    levels.append(level[0])

cabin_df['Deck'] = levels    
```

```python
#Preview new dataframe
cabin_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>person</th>
      <th>Alone</th>
      <th>Survior</th>
      <th>Survivor</th>
      <th>Deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>female</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>male</td>
      <td>Alone</td>
      <td>no</td>
      <td>no</td>
      <td>E</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
      <td>child</td>
      <td>With Family</td>
      <td>yes</td>
      <td>yes</td>
      <td>G</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
      <td>female</td>
      <td>Alone</td>
      <td>yes</td>
      <td>yes</td>
      <td>C</td>
    </tr>
  </tbody>
</table>



```python
#Visualize deck vs survival rate
sns.factorplot('Deck','Survived',data=cabin_df,order=['A','B','C','D','E','F','G'])
```




​    
![png](output_67_2.png)
​    


See no trend in mean of survival rate based on the deck passenger stayed at on titanic. Do see a trend in higher standard deviation of survival rate in lower class deck of "F" and "G" which we have already see that 3rd class passengers were less likely to survive


```python
#Visualize deck survival rate with gender seperation
sns.factorplot('Deck','Survived',hue = 'Sex',data=cabin_df,order=['A','B','C','D','E','F','G'])
```




​    
![png](output_69_2.png)
​    


Once again see lower survival rates of males on board.  Interesting to see no males on the "G" deck which may be related to null values in dataset


```python
#Visualization of survival rate based on deck with class seperation
sns.factorplot('Deck','Survived',hue = 'Pclass',data=cabin_df,order=['A','B','C','D','E','F','G'])
```




​    
![png](output_71_2.png)
​    


Interesting to see the higher survival rate on the "E" deck.  Maybe this was the deck closest to the life boats.

## Did having a family member increase the odd of surviving?


```python
#Survival rate of passenger alone vs with family
sns.factorplot('Alone','Survived',data=titanic_df)
```




​    
![png](output_74_2.png)
​    


Significantly higher survival rate of passengers with family. Wonder if class plays a part into this 


```python
#Include class in visualization
sns.factorplot('Alone','Survived',hue = 'Pclass',data=titanic_df)
```




​    
![png](output_76_2.png)
​    


Once again see the lower survival rate of lower class passenger but little changes in trend based on being with family or alone


```python
#Visualize with gender seperation
sns.factorplot('Alone','Survived',hue = 'Sex',data=titanic_df)
```




​    
![png](output_78_2.png)
​    


Interesting, we see that females traveling alone actually had a higher survival rate than with family but males had a lower.

## Conclusion

We have seen various trends in this data exploration and analysis. We saw that there were more males on board the titanic with majority of passengers embarking from Southhampton.  We also saw that older passengers tended to stay in higher class.

The major trends that stick out in the survivor rate of passengers were:

1. Females and children were more likely to survive than males.
2. Higher class passengers were more likely to surive than lower class passengers
3. Older females were more likley to survive than younger females while the oppposite was true for males.
4. Passengers traveling with family were more likely to survive expect for older females who were more likley with surive if they traveled alone.
5. Passengers on deck "E" appeared to have the highest survival rate of all the decks
