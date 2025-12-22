
import asyncio
import sys
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

# Mock the entire app.config module before importing anything else
sys.modules["app.config"] = MagicMock()
sys.modules["app.tracing"] = MagicMock()
sys.modules["app.services.conversation_logger"] = MagicMock()

# Setup correct return values for mocked modules
sys.modules["app.tracing"].get_run_callbacks.return_value = ([], {})
sys.modules["app.tracing"].get_current_run_id.return_value = "run-123"

# Now import
from app.agents.orchestrator import OrchestratorAgent

async def verify_model_selection():
    print("🧪 Verifying model selection...")
    
    # Mock settings
    with patch("app.agents.base_agent.get_settings") as mock_settings:
        mock_settings.return_value.google_api_key = "fake-key"
        
        # Mock ChatGoogleGenerativeAI in base_agent module
        with patch("app.agents.base_agent.ChatGoogleGenerativeAI") as MockLLM:
            # Setup mock instance
            mock_llm_instance = MagicMock()
            mock_llm_instance.bind_tools.return_value = mock_llm_instance
            mock_llm_instance.ainvoke.return_value = AIMessage(content="Test response")
            # Make sure it returns a dict-like thing if bind_tools is called
            MockLLM.return_value = mock_llm_instance

            # We also need to patch it in orchestrator if it uses it directly?
            # Orchestrator uses self._create_llm which is inherited from BaseAgent.
            # BaseAgent uses ChatGoogleGenerativeAI.
            
            # Prevent ContextManager from doing anything real
            with patch("app.agents.base_agent.ContextManager") as MockContextManager:
                 mock_cm = MagicMock()
                 mock_cm.manage_context.side_effect = lambda msgs: msgs # just return messages
                 MockContextManager.return_value = mock_cm

                 with patch("app.agents.orchestrator.initialize_agent_registry"):
                    orchestrator = OrchestratorAgent()
                    
                    # 1. Test Default
                    print("1️⃣ Testing default model...")
                    try:
                        async for event in orchestrator.stream("Hello"):
                            # Consume generator
                            pass
                    except Exception as e:
                        print(f"❌ Error in stream: {e}")
                        import traceback
                        traceback.print_exc()

                    calls = [call.kwargs.get('model') for call in MockLLM.call_args_list]
                    print(f"   LLM calls so far: {calls}")
                    
                    # We expect 'gemini-2.5-flash' (default)
                    if "gemini-2.5-flash" in calls:
                        print("✅ Default model verified")
                    else:
                        print("❌ Default model NOT found")

                    MockLLM.reset_mock()
                    
                    # 2. Test Custom
                    print("2️⃣ Testing custom model: gemini-3-pro-preview...")
                    try:
                        async for event in orchestrator.stream("Hello", model_name="gemini-3-pro-preview"):
                            pass
                    except Exception as e:
                        print(f"❌ Error in stream: {e}")
                    
                    calls = [call.kwargs.get('model') for call in MockLLM.call_args_list]
                    print(f"   LLM calls: {calls}")
                    
                    if "gemini-3-pro-preview" in calls:
                         print("✅ Custom model verified")
                    else:
                         print("❌ Custom model NOT found")

if __name__ == "__main__":
    try:
        asyncio.run(verify_model_selection())
    except KeyboardInterrupt:
        pass
