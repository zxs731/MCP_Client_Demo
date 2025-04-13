# server.py
from mcp.server.fastmcp import FastMCP
# Create an MCP server
#mcp = FastMCP("MCP_Generate_SQL")
mcp = FastMCP(
    name="MCP_Generate_SQL",
    host="0.0.0.0",
    port=8000,
    description="SQL expert",
    sse_path='/sse'
)

#@mcp.tool()
def hello():
    """
    reply user's hello
    
    """
    return "你好！我是你的Smart Assistant。"
    
@mcp.tool()
def get_tables_schema_information():
    """
    get the tables schema information
    
    """
    infor ="""
Table Name: [Dynamics].[msdyn_workorder]\r\nDescription: 订单表，里面存储工单数据
TABLE [Dynamics].[msdyn_workorder]
(  
    id UNIQUEIDENTIFIER PRIMARY KEY NOT NULL, -- 唯一主键
    mcs_homeccname varchar(200), -- 维修工厂。
    msdyn_name varchar(8), -- 工单号。
    msdyn_addressname varchar(8000), -- 地址。
    mcs_internalmemo varchar(2000), -- 备注或描述。
    msdyn_country varchar(8000), -- 国家。
    mcs_totalaftertax,--税后价格
)
"""
    return infor



@mcp.tool()       
def execute_sql_query(sql:str)-> str:
    """
    execute sql query sentance of sql server. 请考虑由于数据量大，每个查询语句请加入top xx限制查询个数。
    Table Name: [Dynamics].[msdyn_workorder]\r\nDescription: 订单表，里面存储工单数据
    TABLE [Dynamics].[msdyn_workorder]
    (  
        id UNIQUEIDENTIFIER PRIMARY KEY NOT NULL, -- 唯一主键
        mcs_homeccname varchar(200), -- 维修工厂。
        msdyn_name varchar(8), -- 工单号。
        msdyn_addressname varchar(8000), -- 地址。
        mcs_internalmemo varchar(2000), -- 备注或描述。
        msdyn_country varchar(8000), -- 国家。
        mcs_totalaftertax,--税后价格
    )
    return: 
       sql query
    """
    #import pyodbc  

    # 定义连接字符串  
    server = 'xxx'  # SQL Server的名称  
    database = 'xxx'    # 数据库名称  
    driver = '{ODBC Driver 17 for SQL Server}'  # ODBC Driver的名称 
    user='xxx'
    password='xxx'
    # 创建连接字符串，使用Windows集成验证  
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password}'  
    retVal=''
    '''
    # 连接到数据库  
    try:  
        connection = pyodbc.connect(connection_string)  
        print("成功连接到SQL Server!")  
        # 在这里可以执行查询  
        cursor = connection.cursor()  
        cursor.execute(sql)  # 替换为你的表名  
        
        # 获取列名  
        column_names = [column[0] for column in cursor.description]  
        
        rows = cursor.fetchall()  
        
          
        if rows:  
            for row in rows: 
                for col_name, col_value in zip(column_names, row):  # 使用 zip 将列名和列值配对  
                    retVal+=f"{col_name}: {col_value}\r\n"  # 打印列名和列值  
                retVal+="\n"  # 分隔符，用于区分不同的行 
        
    else:  
        print("没有返回任何数据。")  
    except Exception as e:  
        print("连接失败:", e)  
    # 记得在完成后关闭连接  
    finally:  
        if connection:  
            connection.close()  
    '''
    retVal='''
Id: 06afb02d-6610-f011-998a-6045bd68d324
mcs_homeccname: 驻马店维修长
msdyn_name: 12345678
msdyn_addressname: 
mcs_internalmemo:前保险杠损坏
msdyn_country:中国
mcs_totalaftertax:999    
'''
    return retVal
    
if __name__ == "__main__":
   print("Server running")
   #mcp.run(transport='stdio')
   mcp.run(transport='sse')


