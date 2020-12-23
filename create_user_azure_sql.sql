/******* Run This code on the Database you want to give the access ******/

-- The user database where you want to give the access

CREATE USER mlflow_user
FOR LOGIN mlflow_user
WITH DEFAULT_SCHEMA = dbo
GO

-- Add user to the database roles you want
EXEC sp_addrolemember N'db_owner', N'mlflow_user'
GO