from pyspark import SparkConf, SparkContext

sc=SparkContext("local","Simple App")

testFile = "D:\Yaoming\Learn\Python\Pyspark\data\VS14MORT.txt"
testData = sc.textFile(testFile).cache()
numAs = testData.filter(lambda s: 'a' in s).count()
numBs = testData.filter(lambda s: 'b' in s).count()
print ("Lines with a: %i, lines with b: %i" % (numAs, numBs))

print (testData.count())

sc.stop()

sc=SparkContext()

data=sc.parallelize([('Amber',22),('Alfred',23),('Skye',24),('Albert',12),('Amber',9)])
print (data.count())

#reading from file
data_from_file = sc.textFile("D:\Yaoming\Learn\Python\Pyspark\data\VS14MORT.txt",4)
data_from_file.take(1)

###########################################################################
#   Example codes from the book Packt Spark 2.0 for Beginners
###########################################################################

#page 46 from packt Spark 2.0 for Beginners

from decimal import Decimal
from pyspark import SparkConf, SparkContext

sc=SparkContext()

acTransList = ["SB10001,1000", "SB10002,1200", "SB10003,8000",
"SB10004,400", "SB10005,300", "SB10006,10000", "SB10007,500", "SB10008,56",
"SB10009,30","SB10010,7000", "CR10001,7000", "SB10002,-10"]

acTransRDD = sc.parallelize(acTransList)

goodTransRecords = acTransRDD.filter(lambda trans: Decimal(trans.split(",")[1]) > 0).filter(lambda trans:(trans.split(",")[0]).startswith('SB') == True)
goodTransRecords.collect()

highValueTransRecords = goodTransRecords.filter(lambda trans: Decimal(trans.split(",")[1]) > 1000)
highValueTransRecords.collect()

highValueTransRecords = acTransRDD.filter(lambda trans: Decimal(trans.split(",")[1])>1000)
highValueTransRecords.collect()

badAmountLambda = lambda trans: Decimal(trans.split(",")[1]) <= 0
badAcNoLambda = lambda trans: (trans.split(",")[0]).startswith('SB') ==False

badAmountRecords = acTransRDD.filter(badAmountLambda)
badAmountRecords.collect()
badAccountRecords = acTransRDD.filter(badAcNoLambda)
badAccountRecords.collect()

badTransRecords = badAmountRecords.union(badAccountRecords)
badTransRecords.collect()

sumAmounts = goodTransRecords.map(lambda trans: Decimal(trans.split(",")[1])).reduce(lambda a,b : a+b)
sumAmounts

maxAmount = goodTransRecords.map(lambda trans: Decimal(trans.split(",")[1])).reduce(lambda a,b : a if a > b else b)
maxAmount

minAmount = goodTransRecords.map(lambda trans: Decimal(trans.split(",")[1])).reduce(lambda a,b : a if a < b else b)
minAmount

combineAllElements = acTransRDD.flatMap(lambda trans: trans.split(","))
combineAllElements.collect()

allGoodAccountNos = combineAllElements.filter(lambda trans: trans.startswith('SB') == True)
allGoodAccountNos.distinct().collect()

########################## Chap 3 Spark SQL #####################################

# Page 75

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

#spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()

# Creation of the list from where the RDD is going to be created
acTransList = ["SB10001,1000", "SB10002,1200", "SB10003,8000",
"SB10004,400", "SB10005,300", "SB10006,10000", "SB10007,500", "SB10008,56",
"SB10009,30","SB10010,7000", "CR10001,7000", "SB10002,-10"]

sc=SparkContext()

# Create the DataFrame
acTransDF = sc.parallelize(acTransList).map(lambda trans: trans.split(","))\
.map(lambda p: Row(accNo=p[0], tranAmount=float(p[1])))

hasattr(acTransDF, "toDF")

spark = SparkSession(sc)
hasattr(acTransDF, "toDF")

acTransDF = sc.parallelize(acTransList).map(lambda trans: trans.split(","))\
.map(lambda p: Row(accNo=p[0], tranAmount=float(p[1])))

# Convert to DataFrame for using it in SQL
df=acTransDF.toDF()

# Print the structure of the DataFrame
df.printSchema()

# Show the first few records of the DataFrame
df.show()

# register temporary dataframe
df.registerTempTable("temp")

# Use SQL to create another DataFrame containing the good transaction records
goodTransRecords = spark.sql("SELECT accNo,  tranAmount from temp WHERE accNo like 'SB%' AND tranAmount >0")

# Show the first few records of the DataFrame
goodTransRecords.show()

# Use SQL to create another DataFrame containing the high value transaction records
goodTransRecords.registerTempTable("goodtrans")
highValueTransRecords = spark.sql("SELECT accNo, tranAmount FROM goodtrans WHERE tranAmount > 1000")

# Show the first few records of the DataFrame
highValueTransRecords.show()

# Use SQL to create another DataFrame containing the bad account records
badAccountRecords = spark.sql("SELECT accNo, tranAmount FROM temp WHERE accNo NOT like 'SB%'")

# Show the first few records of the DataFrame
badAccountRecords.show()

# Use SQL to create another DataFrame containing the bad amount records
badAmountRecords = spark.sql("SELECT accNo, tranAmount FROM temp WHERE tranAmount < 0")

# Show the first few records of the DataFrame
badAmountRecords.show()

# Do the union of two DataFrames and create another DataFrame
badTransRecords=badAccountRecords.union(badAmountRecords)

#Show the first few records of the DataFrame
badTransRecords.show()

# Calculate the sum
sumAmount = spark.sql("SELECT sum(tranAmount) as sum FROM goodtrans")

# Show the first few records of the DataFrame
sumAmount.show()

# Calculate the maximum
maxAmount = spark.sql("SELECT max(tranAmount) as max FROM goodtrans")

# Show the first few records of the DataFrame
maxAmount.show()

# Calculate the minimum
minAmount = spark.sql("SELECT min(tranAmount)as min FROM goodtrans")

# Show the first few records of the DataFrame
minAmount.show()

# Use SQL to create another DataFrame containing the good account numbers
goodAccNos = spark.sql("SELECT DISTINCT accNo FROM temp WHERE accNo like 'SB%' ORDER BY accNo")

# Show the first few records of the DataFrame
goodAccNos.show()

# Calculate the sum using mixing of DataFrame and RDD like operations
sumAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a+b)

sumAmountByMixing

# Calculate the maximum using mixing of DataFrame and RDD like operations
maxAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a if a > b else b)
maxAmountByMixing

# Calculate the minimum using mixing of DataFrame and RDD like operations
minAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a if a < b else b)
minAmountByMixing

# Use SQL to create another DataFrame containing the account summary records
acSummary = spark.sql("SELECT accNo, sum(tranAmount) as transTotal FROM temp GROUP BY accNo ORDER BY accNo")

# Show the first few records of the DataFrame
acSummary.show()

# Create the DataFrame using API for the account summary records
acSummaryViaDFAPI=df.groupBy("accNo").agg({"tranAmount":"sum"})\
.selectExpr("accNo","`sum(tranAmount)` as transTotal")

acSummaryViaDFAPI.show()

sc.stop()

##################### Understanding multi-datasource joining with SparkSQL page 91    ####################

from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

sc=SparkContext()
spark = SparkSession(sc)

# Creation of the list from where the RDD is going to be created
AcMaster = Row('accNo', 'firstName', 'lastName')
AcBal = Row('accNo', 'balanceAmount')

acMasterList = ["SB10001,Roger,Federer","SB10002,Pete,Sampras",
"SB10003,Rafael,Nadal","SB10004,Boris,Becker", "SB10005,Ivan,Lendl"]
acBalList = ["SB10001,50000", "SB10002,12000","SB10003,3000",
"SB10004,8500", "SB10005,5000"]

# Create the DataFrame
acMasterDF_temp = sc.parallelize(acMasterList).map(lambda trans:\
trans.split(",")).map(lambda r: AcMaster(*r))

acMasterDF=acMasterDF_temp.toDF()

acMasterDF.printSchema()

acBalDF_temp = sc.parallelize(acBalList).map(lambda trans:\
trans.split(",")).map(lambda r: AcBal(r[0], float(r[1])))

acBalDF=acBalDF_temp.toDF()
acBalDF.show()

# Persist the data of the DataFrame into a Parquet file
acMasterDF.write.parquet("python.master.parquet")

# Persist the data of the DataFrame into a JSON file
acBalDF.write.json("pythonMaster.json")

# Read the data into a DataFrame from the Parquet file
acMasterDFFromFile = spark.read.parquet("python.master.parquet")

# Register temporary table in the DataFrame for using it in SQL
acMasterDFFromFile.createOrReplaceTempView("master")

# Register temporary table in the DataFrame for using it in SQL
acBalDFFromFile = spark.read.json("pythonMaster.json")

# Register temporary table in the DataFrame for using it in SQL
acBalDFFromFile.createOrReplaceTempView("balance")

# Show the first few records of the DataFrame
acMasterDFFromFile.show()

# Show the first few records of the DataFrame
acBalDFFromFile.show()

# Use SQL to create another DataFrame containing the account detail records
acDetail = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount \
                     FROM master, balance WHERE master.accNo = balance.accNo \
                     ORDER BY balanceAmount DESC")

# Show the first few records of the DataFrame
acDetail.show()

# Create the DataFrame using API for the account detail records
acDetailFromAPI = acMasterDFFromFile.join(acBalDFFromFile, acMasterDFFromFile.accNo ==\
                                          acBalDFFromFile.accNo).sort(acBalDFFromFile.balanceAmount,\
                                          ascending=False).select(acMasterDFFromFile.accNo,\
                                          acMasterDFFromFile.firstName, acMasterDFFromFile.lastName,\
                                          acBalDFFromFile.balanceAmount)
# Show the first few records of the DataFrame
acDetailFromAPI.show()

# Use SQL to create another DataFrame containing the top 3 account detail records
acDetailTop3 = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount\
                         FROM master, balance WHERE master.accNo = balance.accNo \
                         ORDER BY balanceAmount DESC").limit(3)
# Show the first few records of the DataFrame
acDetailTop3.show()

############ Understanding Data Catalogs page 103   ############

#Get the catalog object from the SparkSession object
catalog = spark.catalog
#Get the list of databases and their details.
catalog.listDatabases()

#Display the details of the tables in the database
catalog.listTables()

