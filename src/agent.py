import re
from src.llms.base_llm import BaseLLM
from src.tools.base_tool import BaseTool


class Agent:
    """
    Bounded ReAct-style agent: on each step, asks the LLM to either
    call a tool or give a final answer. Loop is capped at max_steps
    so a confused LLM can't loop forever — this directly satisfies
    the Chapter 3 checklist: "bounded/controllable behavior."

    Clear handoff: the LLM only ever DECIDES (produces text saying
    which tool + input). This class is the only thing that ACTS
    (actually executes tool.run()). The LLM never executes code itself.
    """

    def __init__(self, llm: BaseLLM, tools: list[BaseTool], max_steps: int = 5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps

    def _build_prompt(self, question: str, history: str) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )
        return (
            "You are an agent that answers questions by using tools.\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "Respond in EXACTLY one of these two formats:\n"
            "Action: <tool_name> | <tool_input>\n"
            "Final Answer: <your answer>\n\n"
            f"History so far:\n{history}\n"
            f"Question: {question}\n"
        )

    def run(self, question: str) -> str:
        history = ""

        for step in range(self.max_steps):
            prompt = self._build_prompt(question, history)
            response = self.llm.generate(prompt).strip()

            # Decide: parse whether the LLM chose an action or a final answer
            final_match = re.match(r"Final Answer:\s*(.+)", response, re.DOTALL)
            if final_match:
                return final_match.group(1).strip()

            action_match = re.match(r"Action:\s*(\w+)\s*\|\s*(.+)", response, re.DOTALL)
            if not action_match:
                # LLM didn't follow format — surface this rather than guessing
                history += f"\n[Unparseable response: {response}]\n"
                continue

            tool_name, tool_input = action_match.group(1), action_match.group(2).strip()

            # Act: this class executes the tool, not the LLM
            if tool_name not in self.tools:
                observation = f"Error: unknown tool '{tool_name}'."
            else:
                observation = self.tools[tool_name].run(tool_input)

            history += f"\nAction: {tool_name} | {tool_input}\nObservation: {observation}\n"

        return "Agent stopped: max steps reached without a final answer."