import asyncio
from contextlib import AsyncExitStack
import json
from dotenv import load_dotenv 
import os
import sys
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import AsyncAzureOpenAI  
import httpx 
load_dotenv("./azureopenai.env")  
model=os.getenv("model")
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.sessions={}
        self.exit_stack = AsyncExitStack()
        self.tools=[]
        self.messages=[]
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.environ["base_url"],  
            api_key=os.environ["api_key"],  
            api_version="2024-05-01-preview",
            http_client=httpx.AsyncClient(verify=False)   
        )


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def connect_to_server(self):
        with open("mcp_server_config.json", "r") as f:
            config = json.load(f)
            print(config["mcpServers"])  
        conf=config["mcpServers"]
        self.tools=[]
        for key in conf.keys():  
            v = conf[key] 
            session = None
            if "baseUrl" in v and v['isActive']:
                server_url = v['baseUrl']
                sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
                write,read = sse_transport
                session = await self.exit_stack.enter_async_context(ClientSession(write,read))
            elif "command" in v  and v['isActive']:
                command = v['command']
                args=v['args']
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=None
                )
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio1, write1 = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio1, write1))

            if session:
                await session.initialize()  
                response = await session.list_tools()  
                tools = response.tools 
                print("tools load ......")
                for tool in tools:  
                    self.sessions[tool.name] = session
                    print(tool.name)
                self.tools += tools
                print("tools loaded done!")
            else:
                print("No session!")
                
    async def run_conversation(self,messages,tools,think_handle=None,content_handle=None):
        # Step 1: send the conversation and available functions to the model
        
        response_message = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True
        )
        content=''
        reasoning_content=''
        function_list=[]
        index=0
        async for chunk in response_message:
            if chunk and len(chunk.choices)>0:
                chunk_message =  chunk.choices[0].delta
                #print(chunk_message)
                if chunk_message.content:
                    content+=chunk_message.content
                    #print(chunk_message.content,end="")
                    if content_handle:
                        content_handle(chunk_message.content)
                  
                if chunk_message.tool_calls:
                    
                    for tool_call in chunk_message.tool_calls:
                        
                        if len(function_list)<tool_call.index+1:
                            function_list.append({'name':'','args':'','id':tool_call.id})
                        if tool_call and tool_call.function.name:
                            function_list[tool_call.index]['name']+=tool_call.function.name
                        if tool_call and tool_call.function.arguments:
                            function_list[tool_call.index]['args']+=tool_call.function.arguments
                            
    
        #print(function_list)
        
        if len(function_list)>0:
            findex=0
            tool_calls=[]
            temp_messages=[]
            for func in function_list:
                function_name = func["name"]
                #print(function_name)
                function_args = func["args"]
                function_args = json.loads(function_args)
                toolid=func["id"]
                if function_name !='':
                    #print(f'⏳Call {function_name}...')
                    # 执行工具调用
                    function_response = await self.sessions[function_name].call_tool(function_name, function_args)
                    print(f"MCP: [Calling tool {function_name} with args {function_args}]")
                    #print(f'⏳Call internal function done! ')
                    #print("执行结果：")
                    #print(function_response.content)
                    tool_calls.append({"id":toolid,"function":{"arguments":func["args"], "name":function_name}, "type":"function","index":findex})
                    
                    temp_messages.append(
                        {
                        "tool_call_id": toolid,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response.content,
                        }
                    )
                    if think_handle:
                        think_handle(f"Call {function_name}({function_args})")
                    #print(messages)
                    findex+=1
                    
            messages.append({
                        "role":"assistant",
                        "content":content,
                        "tool_calls":tool_calls,
                    })
            for m in temp_messages:
                messages.append(m)
            #print("-------------------------")
            #print(messages)  
            
            return await self.run_conversation(messages,tools,think_handle,content_handle)
        elif content!='':# and messages[-1]["role"]!="tool":
            messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
            #print(messages)
            return messages[-1]
            
    async def process_query(self, query: str,content_handle=None,think_handle=None) -> str:
        """使用 LLM 和 MCP 服务器提供的工具处理查询"""
        self.messages=self.messages+[
            {
                "role": "user",
                "content": query
            }
        ]
        messages =self.messages[-20:]
        
        #print(messages)
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in self.tools]
        replyMessage =await self.run_conversation(messages,available_tools,think_handle,content_handle)
        #print("=================")
        #print(replyMessage)
        self.messages=self.messages+[replyMessage]
        return replyMessage["content"]

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() == 'quit':
                    break
                print("\nAI: ")
                response = await self.process_query(query,content_handle=lambda x:print(x,end=""))
                #print("\nAI: " + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
        print("AI: Bye! See you next time!")

if __name__ == "__main__":
    asyncio.run(main())

#uv run client.py 启动客户端
