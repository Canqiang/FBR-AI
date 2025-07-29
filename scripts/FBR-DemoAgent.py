import json
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import random
import os
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

# ========== 初始化 LLM ==========
# Load Azure OpenAI Config
def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found. Please create it.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = config["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = config["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = config["AZURE_OPENAI_API_VERSION"]
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    temperature=0
)

# llm = ChatOpenAI(
#     base_url='https://api.openai-proxy.org/v1',
#     model = "gpt-4o-mini",
#     api_key='sk-Fmiw2WNajQ7fkU6thUpMqKEUCTk2D1r1JGmRWfv8k7p8s1pu',
# )


# ========== 工具定义 ==========
def forecast_sales_tool(query: str = "") -> str:
    """
    Forecast next 7 days of sales using Prophet (mock data).
    Returns JSON with date and predicted_sales.
    """
    dates = pd.date_range(datetime.today() - timedelta(days=30), datetime.today())
    sales = [random.randint(80, 150) for _ in range(31)]
    df = pd.DataFrame({"ds": dates, "y": sales})

    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future).tail(7)[["ds", "yhat"]]

    forecast_data = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "predicted_sales": int(row["yhat"])}
        for _, row in forecast.iterrows()
    ]
    return json.dumps(forecast_data, indent=2)

forecast_tool = Tool(
    name="SalesForecast",
    func=forecast_sales_tool,
    description="Predict sales for the next 7 days. Returns JSON with date and predicted_sales."
)

def generate_strategy_tool(forecast_json: str) -> str:
    """
    Generate restocking and promotion strategies in strict JSON format,
    including reasons for each action.
    """
    prompt = f"""
    You are a supply chain strategy AI.
    Based on the forecast data below, create a JSON plan:
    {forecast_json}

    Rules:
    - Use ONLY the dates from the forecast data.
    - Output ONLY a JSON array.
    - Each item must include:
        "date": a date from forecast data.
        "action": e.g., "Restock 100 units" or "Launch promotion".
        "reason": a short explanation for why this action is recommended (max 1 sentence).
    - No text outside JSON.
    """

    response = llm.predict(prompt).strip()
    try:
        parsed = json.loads(response)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        # Attempt to repair JSON
        fix_prompt = f"""
        The following text should be valid JSON but is not:
        {response}
        Fix it into a valid JSON array of objects with 'date', 'action', and 'reason'.
        Use only the dates from forecast data:
        {forecast_json}
        """
        fixed = llm.predict(fix_prompt).strip()
        return fixed


strategy_tool = Tool(
    name="StrategyGenerator",
    func=generate_strategy_tool,
    description="Generate restocking and promotion recommendations in strict JSON."
)

# ========== Prompt + Memory ==========
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
    You are an intelligent assistant that name is "FBR-Assistant".
    - If user asks about sales prediction or restocking strategy, use the tools SalesForecast and StrategyGenerator.
    - If user is chatting, answer directly without calling any tool.
    - Always return results in English.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ========== 创建 Agent ==========
functions_agent = create_openai_functions_agent(
    llm=llm,
    tools=[forecast_tool, strategy_tool],
    prompt=prompt
)

agent = AgentExecutor(
    agent=functions_agent,
    tools=[forecast_tool, strategy_tool],
    verbose=True,
    memory=memory
)

# ========== 对话入口 ==========
def chat_with_agent():
    print("=== Decision Agent (with Memory) ===")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break
        try:
            response = agent.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_agent()