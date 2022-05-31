import mysql.connector
import pandas as pd
# try:

#   mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="admin",
#     database="loop_template_plant",
#     port = '3306'
#   )

#   mycursor = mydb.cursor()
#   mySql_insert_query  = """INSERT INTO model_Details (model_id,plant_name,model_type,model_path, is_main, model_accurary) 
#                             VALUES 
#                             (1,'Satorp', 'Loop level template', '/models/Satorp', 1,99.56) """

#   mycursor.execute(mySql_insert_query)
#   mydb.commit()
#   print(mycursor.rowcount, "Record inserted successfully into model_details table")
#   mycursor.close()

# except mysql.connector.Error as error:
#     print("Failed to insert record into model_details table {}".format(error))

# finally:
#     if mydb.is_connected():
#         mydb.close()
#         print("MySQL connection is closed")

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="loop_template_plant",
    port = '3306'
  )
mycursor = mydb.cursor()
sql ="select model_path,model_accurary from  model_Details where plant_name = %s and is_main = %s"

adr = ('Satorp',True )
mycursor.execute(sql, adr)
myresult = mycursor.fetchall()

feature_list =[]
dependent_feature = []
print(myresult[0][0])
print(myresult[0][1])


if mydb.is_connected():
  mydb.close()


