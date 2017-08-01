############## Packt Spark 2.0 for Beginners ################################
############# Chap 5 Spark Data Analysis with Python #########################

#Import all the required libraries

from pyspark.sql import Row
import matplotlib.pyplot as plt
import numpy as np
import pylab as P
from pyspark  import SparkConf, SparkContext
from pyspark.sql import SparkSession

sc=SparkContext()
spark=SparkSession(sc)
plt.rcdefaults()

# TODO - The following location has to be changed to the appropriate data file location

dataDir ="D:/Yaoming/Learn/Python/Pyspark/data/ml-100k/"

# Create the DataFrame of the user dataset
lines = sc.textFile(dataDir + "u.user")
lines.collect()
splitLines = lines.map(lambda l: l.split("|"))
usersRDD = splitLines.map(lambda p: Row(id=p[0], age=int(p[1]), gender=p[2], occupation=p[3], zipcode=p[4]))

hasattr(usersRDD, "toDF")

usersDF=spark.createDataFrame(usersRDD)
usersDF.createOrReplaceTempView("users")
usersDF.show()

# Create the DataFrame of the user dataset with only one column age
ageDF = spark.sql("SELECT age FROM users")
ageList=ageDF.rdd.map(lambda p: p.age).collect()
ageDF.describe().show()

# Age distribution of the users
plt.hist(ageList)
plt.title("Age distribution of the users\n")
plt.xlabel("Age")
plt.ylabel("Number of users")
plt.show(block=False)


# Draw a density plot
from scipy.stats import gaussian_kde
density = gaussian_kde(ageList)
xAxisValues = np.linspace(0,100,1000)
density.covariance_factor = lambda : 0.1
density._compute_covariance()
plt.title("Age density plot of the users\n")
plt.xlabel("Age")
plt.ylabel("Density")
plt.plot(xAxisValues, density(xAxisValues))
plt.show(block=False)

# The following example demonstrates the creation of multiple diagrams in one figure
# There are two plots on one row
# The first one is the histogram of the distribution
# The second one is the boxplot containing the summary of the distribution

plt.subplot(121)
plt.hist(ageList)
plt.title("Age distribution of the users\n")
plt.xlabel("Age")
plt.ylabel("Number of users")
plt.subplot(122)
plt.title("Summary of distribution\n")
plt.xlabel("Age")
plt.boxplot(ageList, vert=False)
plt.show(block=False)

#Bar chart
occupationsTop10 = spark.sql("SELECT occupation, count(occupation) as usercount\
                             FROM users \
                             GROUP BY occupation \
                             ORDER BY usercount DESC LIMIT 10")
occupationsTop10.show()
occupationsTop10Tuple = occupationsTop10.rdd.map(lambda p: (p.occupation,p.usercount)).collect()
occupationsTop10List, countTop10List = zip(*occupationsTop10Tuple)
occupationsTop10Tuple
# Top 10 occupations in terms of the number of users having that occupation who have rated movies
y_pos = np.arange(len(occupationsTop10List))
plt.barh(y_pos, countTop10List, align='center', alpha=0.4)
plt.yticks(y_pos, occupationsTop10List)
plt.xlabel('Number of users')
plt.title('Top 10 user types\n')
plt.gcf().subplots_adjust(left=0.15)
plt.show(block=False)

#Stacked bar chart
occupationsGender = spark.sql("SELECT occupation, gender FROM users")
occupationsGender.show()
occCrossTab = occupationsGender.stat.crosstab("occupation", "gender")
occCrossTab.show()
occupationsCrossTuple = occCrossTab.rdd.map(lambda p: (p.occupation_gender,p.M, p.F)).collect()
occList, mList, fList = zip(*occupationsCrossTuple)
N = len(occList)
ind = np.arange(N) # the x locations for the groups
width = 0.75 # the width of the bars
p1 = plt.bar(ind, mList, width, color='r')
p2 = plt.bar(ind, fList, width, color='y', bottom=mList)
plt.ylabel('Count')
plt.title('Gender distribution by occupation\n')
plt.xticks(ind + width/2., occList, rotation=90)
plt.legend((p1[0], p2[0]), ('Male', 'Female'))
plt.gcf().subplots_adjust(bottom=0.25)
plt.show(block=False)




