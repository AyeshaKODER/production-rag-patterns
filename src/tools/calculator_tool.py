from src.tools.base_tool import BaseTool


class CalculatorTool(BaseTool):
    """
    Simple arithmetic tool. Exists mainly to prove the agent can choose
    between multiple distinct tools based on the question, not just
    always call the one tool it has.
    """

    name = "calculator"
    description = (
        "Use this for arithmetic calculations, e.g. sums, differences, "
        "percentages. Input should be a plain math expression like '85 * 30'."
    )

    def run(self, input_str: str) -> str:
        try:
            # Restricted eval: only digits, operators, parens, decimal points
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in input_str):
                return "Error: invalid characters in expression."
            return str(eval(input_str))
        except Exception as e:
            return f"Error: {e}"