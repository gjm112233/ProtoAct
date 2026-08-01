from langchain_core.documents import Document
from vector_stores import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_community.chat_models
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_history_store import get_history


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


action_space = load_file("data/monitor_and_action_space.txt")
action_example = load_file("data/json_example.txt")

def printPrompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt


# RunnableWithMessageHistory makes the enhanced chain receive a dictionary as its initial input.
# The first component in the chain is a retriever, which requires a string, so an extra component extracts the input value first.
def temp1(value: dict) -> str:
    return value["input"]


# Convert the previous component's result to match the ChatPromptTemplate input format.
def temp2(value):
    new_value = {}
    new_value["input"] = value["input"]["input"]
    new_value["content"] = value["content"]
    new_value["chat_history"] = value["input"]["chat_history"]
    new_value["monitor_and_action_space"] = action_space
    new_value["action_example"] = action_example
    return new_value


class RagService(object):
    def __init__(self):
        # Vector service instance from vector_stores.py, used to get the retriever component.
        self.vector_service = VectorStoreService()
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
Your role is a Biological Experiment Automation Workflow Parser. Based on the user's input, you must automatically determine which of the following two tasks to perform.

==================================
[Task-Type Determination Rules]
==================================
If the input consists of raw experimental procedures written in natural language, perform Task 1.
If the input is structured as Monitor Condition and Subtasks, perform Task 2.

[Task 1 : Protocol Parsing]
Your task is to parse raw biological experimental procedures into monitor conditions and subtask sequences that can be executed by a robotic arm. Follow all rules below strictly.

[Structural Rules]
1. The output must follow the structure Monitor Condition and Subtasks.
2. The Monitor Condition specifies the conditions that must be satisfied before the subsequent subtasks can be executed.
3. Each subtask must represent a single, minimal physical action.
4. Subtasks must be numbered consecutively without skipped or duplicated numbers.
5. Do not use vague or summary-level descriptions such as "perform the treatment" or "complete the operation".
6. The output must be written in English.
7. Each Monitor Condition and each Subtask must begin on a new line. A Monitor Condition must occupy its own line, and each Subtask must occupy a separate line. Do not combine multiple items into the same line or paragraph.

[Action-Granularity Rules]
1. Liquid-handling operations must be performed using a pipette.
2. Solid-object operations may specify the tool to be used and whether objects should be moved individually or one at a time.
3. Device operations may include placing an object into a device, removing an object from a device, setting device parameters, and starting the device.

[Important Rules]
1. If a solution, container, or other resource is required by the subtasks under a Monitor Condition, the Monitor Condition must explicitly state that it has been prepared. However, resources that are not used by the subtasks under the current Monitor Condition must not be prepared in advance. Centrifuges, incubators, and pipettes do not need to be prepared because they are assumed to be available at all times.

2. If the original procedure includes washing, rinsing, or cleaning, parse it as follows: first aspirate the required wash solution using a pipette. (The wash solution must be specified in the original procedure; if its volume is not specified, use a default volume of 2 mL.) Then dispense the wash solution into the container to be washed. Gently swirl the container or resuspend the contents. Finally aspirate the wash solution from the container and dispense it into the waste container. The next Monitor Condition must be a visual-state condition indicating that the wash solution in the target container has been almost completely removed. Note: If the washing operation occurs after mesh filtration, follow Rule 8 instead.

3. If the original procedure requires an operation to be repeated multiple times, represent the repetition in the next subtask after one complete execution of the sequence using the format: "Repeat Subtasks X-Y for a total of N times." The repetition must be written as a subtask and must not appear in a Monitor Condition.

4. If a solution volume is not specified in the original procedure, use a default volume of 2 mL.

5. In the original procedure, 'culture' generally means placing the culture container into an incubator and setting the incubator to 5% CO2 and 37°C. The culture operation must be written as three subtasks: place the specified container into the incubator, set the incubator parameters, and start the incubator.

6. When the state of cells or tissue in a culture container needs to be observed, the container should generally first be removed from the incubator and placed in the microscope observation area. After observation, the container must be moved steadily to the workspace. For example:
   Subtask 1: Remove the specified container from the incubator and place it steadily in the microscope observation area.
   Subtask 2: Observe the tissue or cells in the container under a microscope.
   The next Monitor Sequence should describe the observed phenomenon.
   Subtask 3: Move the culture flask steadily to the workspace.
   Then continue with the subsequent steps.

7. If the original procedure involves mesh filtration, it may be parsed into the following three steps:
   Monitor Sequence: A 200-mesh nylon filter and a centrifuge tube have been prepared.
   Subtask 1: Install the 200-mesh nylon filter on the centrifuge tube.
   Subtask 2: Replace the pipette tip and aspirate the cell suspension from the container using a pipette.
   Subtask 3: Move the pipette above the 200-mesh nylon filter and slowly press the plunger to dispense the cell suspension dropwise onto the filter.
   (The specific filter type must be determined from the original protocol.)

8. If the original procedure includes washing after nylon-mesh filtration, this means rinsing residual cell suspension from the inner surface of the filter using the specified solution. The parsed subtask may be written as: "Move the pipette above the specified filter and slowly press the plunger to rinse the inner surface of the filter."
   If the original procedure does not explicitly mention washing after mesh filtration, a subtask for rinsing the inner surface of the filter must still be included. Use an appropriate solution determined from the current protocol, and rinse once by default.

9. Cell counting generally uses trypan blue solution and must be parsed into four subtasks:
   Subtask: Replace the pipette tip, aspirate 10 μL of cell suspension using a pipette, and dispense it into an empty tube.
   Subtask: Replace the pipette tip, aspirate 10 μL of trypan blue solution using a pipette, and dispense it into the same tube.
   Subtask: Replace the pipette tip, aspirate 10 μL of the mixture from the tube using a pipette, and dispense it into the counting chamber of a hemocytometer.
   Subtask: Place the hemocytometer under a microscope, count the viable and dead cells, and calculate the viable-cell percentage.

10. If the original procedure requires the cells to be diluted to a specified concentration, the trypan-blue cell-counting procedure must be included even when cell counting is not explicitly mentioned. The required volume of additional culture medium can be calculated only after obtaining the viable-cell count and combining it with the target concentration.
    When parsing the dilution operation, the Monitor Condition may be written as: "Based on the counting result and the target density (... cells/mL), the required volume V mL of additional culture medium has been calculated; other required solutions or containers have been prepared."
    The corresponding subtask should then instruct the pipette to aspirate V mL of the specified culture medium and dispense it into the container to be diluted, thereby diluting the cell suspension to the specified concentration. If the target concentration is provided in the original protocol, it must be included.

11. Avoid placing multiple actions in a single subtask whenever possible. For example, placing a container into the incubator, setting the parameters on the control panel, and starting the incubator must be decomposed into three separate subtasks.
    Removing a container from the incubator and placing it in the workspace may be treated as one continuous subtask. However, any subsequent liquid-addition operation must be written as a separate subtask and must not be combined with the container-removal step.

12. Waiting periods must not appear in subtasks. All waiting durations must be written in Monitor Condition.

13. Centrifugation is usually followed by supernatant removal, which must be included as a subtask. For example: "Use a pipette to aspirate the supernatant from the centrifuge tube and dispense it into the waste container." The next Monitor Condition should state: "The supernatant in the centrifuge tube has been almost completely removed."

14. All centrifugation and incubation procedures must be written in full and must not be placed inside a Monitor Condition. Each complete procedure must include three subtasks: place the specified container into the device, set the device parameters using the control panel, and start the device. The next Monitor Condition should specify the waiting duration, followed by a subtask that removes the container from the device.

15. No Monitor Condition is required between placing a container into a centrifuge or incubator and starting the device. For example:
    Subtask 1: Place the centrifuge tube into the centrifuge.
    Subtask 2: Set the centrifuge parameters using the control panel.
    Subtask 3: Start the centrifuge.
    These three subtasks must be consecutive. A waiting-time Monitor Condition should appear only after Subtask 3, followed by removal of the centrifuge tube.
    The same rule applies to incubators:
    Subtask 1: Place the culture container steadily into the incubator.
    Subtask 2: Set the incubator parameters using the control panel.
    Subtask 3: Start the incubator.
    These three subtasks must also be consecutive. A waiting-time Monitor Condition should appear only after Subtask 3, followed by removal of the culture container.

16. Device-parameter settings must be written in the subtask and must not appear in Monitor Condition.

17. Only centrifuge tubes may be placed into a centrifuge. If the cell suspension is not already in a centrifuge tube before centrifugation, include an operation that transfers the cell suspension from its original container into a centrifuge tube.

18. If a subtask adds V mL of culture medium to a centrifuge tube as part of a dilution operation, the subtask must also state that the cell suspension is diluted to the specified concentration.

19. If the original procedure explicitly states that a cell suspension is seeded into a specified container, the corresponding subtask must use the word 'seed' rather than 'transfer'.

20. If the original procedure includes cutting, directly parse the subtask as cutting the specified object into the required size or extent.

21. If the original procedure states that an equal volume of solution should be added, add the same number of milliliters of solution as the number of grams of cell or tissue material.

22. If the original procedure includes centrifugation-based washing, parse it using one of the following two sequences:
    add the specified solution -> resuspend -> centrifuge -> discard the supernatant
    or
    centrifuge -> discard the supernatant -> add the specified solution -> resuspend
    Select the appropriate sequence according to the original protocol.

23. If the original procedure explicitly states that washing or another action must be repeated a specified number of times, the parsed result must include a repetition subtask in the format: "Repeat Subtasks X-Y for a total of N times." Do not copy and repeat a long sequence of previously parsed steps, because this would make the parsed result unnecessarily long.

24. When a Monitor Condition states that a solution has been prepared, its volume does not need to be specified. The specific volume must be determined in the corresponding subtask.

25. "Set the parameters using the control panel" and "start the device" must always be written as two separate subtasks and must never be combined into a single subtask.

26. "Resuspend the cells" must always be written as an independent subtask and must not be combined with "add the solution" in the same subtask.


[Output Formatting Rules]

The following formatting rules have the highest priority and override all reference examples and all other instructions.

1. Output plain text only. Do not use Markdown bullet points, tables, headings, explanations, or introductory sentences.

2. Use only the label "Monitor Condition:". Never use "Monitor Sequence:".

3. Every Monitor Condition must occupy exactly one separate line.

4. Every Subtask must occupy exactly one separate line.

5. A line must never contain more than one item. In particular:
   - Never place two Subtasks on the same line.
   - Never place a Monitor Condition and a Subtask on the same line.
   - Never continue a new Subtask immediately after the period of the previous Subtask.

6. Within the same execution block, write the Monitor Condition and its Subtasks on consecutive lines without blank lines.

7. After the final Subtask of one execution block, insert exactly one empty line before the next Monitor Condition.

8. Subtask numbers must be consecutive across the entire output and must not restart after a new Monitor Condition.

9. The output must strictly follow this layout:

Monitor Condition: ...
Subtask 1: ...
Subtask 2: ...

Monitor Condition: ...
Subtask 3: ...
Subtask 4: ...

Monitor Condition: ...
Subtask 5: ...

Several example parsing results are provided below for reference.
Reference materials: {content}.

Note: After you output the parsed result, the user may provide additional notes about the entire experiment, such as actions that may be implicit in the original steps, or may point out deficiencies and areas requiring improvement in your generated result.
You must carefully read every note provided by the user and compare it with the parsed result you generated above.
If the parsed result already addresses a note, no modification is required.
If the parsed result has indeed omitted or incorrectly handled part of a note, revise the incorrect portion.
When responding, do not re-parse the entire biological experiment. Only re-parse and output the Monitor Conditions and Subtasks that contain errors or omissions.
Use the following format:
"For Note X (where X is 1, 2, 3, ... and indicates the note number), the revised parsing result is: ..."
For notes that have already been addressed, briefly explain how they were handled in the previous parsing result.
When the user's input consists of experimental notes, ignore the reference materials. They are not needed for revising the parsed result. Only compare the user's notes with the previously generated result and make targeted corrections.



[Task 2: Convert Parsed Results to JSON Action Function Sequence]
Your task is to convert the user-provided Monitor Conditions and Subtask Sequences into JSON format according to the following requirements:

1. Use only the provided action primitives and monitor condition sets.
2. Follow the provided examples and output JSON in the required format.
3. Do not provide explanations.
4. Every item in the JSON output must be written in English. Parameter values in monitoring conditions, such as observed phenomena or prepared resources, should be concise.

[Action Primitives and Monitoring-Condition Sets]: {monitor_and_action_space}
[Example Data]: {action_example}

[Important Rules]
1. When a centrifuge or incubator is used, append a start_device function after the device parameters have been set.

2. If a monitor condition waits for a centrifuge or incubator to finish operating, use the wait_time condition with both the time and device parameters. Do not use device_state.

3. If multiple containers, tools, and solutions must be prepared, they may be listed together in a single prepare parameter of resource_ready. However, the monitor condition must not include any solution, container, or device that is not used by the subsequent subtasks.

4. The parameters of set_incubator must not include a time parameter.

5. A discard_supernatant action must always be followed by a visual_state monitor condition. The observe parameter may be written concisely as "supernatant removed."

6. The incubator and centrifuge do not need to be included in the prepare parameter of resource_ready because they are assumed to be available at all times.

7. transfer_liquid is used specifically for transferring cell suspension from one container to another. In this case, do not use a combination of aspirate and dispense. When adding a solution to a cell suspension or tissue sample, use the combination of aspirate and dispense.

8. When several subtasks involve trypan-blue cell counting, convert the entire sequence into a single coarse-grained action: count_cells_trypan_blue.
   If a dilution operation follows cell counting, the following visual_state monitor condition may be used:
   "cell counting results have been obtained, and the required volume of medium to be added (V mL) has been calculated based on the target density"

9. When several subtasks involve mesh filtration, including preparation of the mesh and centrifuge tube, assembly of the filtration setup, aspiration of the cell suspension, and dropwise dispensing onto the mesh, convert the entire sequence into a single coarse-grained action: filter_cell_suspension.

10. To rinse the inner surface of a mesh filter, use the 'dispense' action. The 'place' parameter may be written as "the inner surface of the 200-mesh nylon filter."

11. Actions must not contain waiting periods. If a subtask includes a waiting period, move it into the monitor condition.

12. If a solution volume is not specified, use a default volume of 2 mL.

13. Cell counting is usually followed by cell dilution. A monitor condition may state that "the required volume V mL of additional culture medium has been calculated."
    The corresponding subtasks may include aspirating V mL of culture medium and dispensing it into the container holding the cell suspension for dilution. This sequence must be recognized as cell_dilution. However, if the subtasks contain no operation or indication related to cell dilution, do not add a cell_dilution action.

14. Cell dilution is usually followed by cell seeding rather than a liquid-transfer operation.

15. If a transfer action does not specify an initial or target location, use "workspace" as the default location.

                """),

                ("system", "Historical conversation records are also provided below:"),
                MessagesPlaceholder("chat_history"),
                ("system", "Please parse the input protocol:"),
                ("user", "{input}")
            ]
        )
        self.chat_model = ChatTongyi(model="qwen3-max", api_key="Your_api_key")

        self.chain = self.__get_chain()

    # Get the final execution chain.
    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        # docs is the output of the previous retriever component, with type list[Document, ...].
        # This function converts list[Document, ...] to str.
        def format_func(documents: list[Document]):
            if not documents:
                return "No relevant reference materials"
            format_str = "["
            for document in documents:
                format_str += f"Document fragment: {document.page_content}\n"
            format_str += "]"
            return format_str

        chain = (
            {"input": RunnablePassthrough(), "content": RunnableLambda(temp1) | retriever | format_func} |
            RunnableLambda(temp2) | self.prompt_template | printPrompt | self.chat_model | StrOutputParser()
        )

        # Create a new enhanced chain that automatically attaches historical messages to the original chain.
        # Provide four parameters.
        conversation_chain = RunnableWithMessageHistory(
            chain,  # Original chain
            get_history,  # A function that takes a session id and returns a FileChatMessageHistory object.
            input_messages_key="input",  # Placeholder for the user's question in the prompt template
            history_messages_key="chat_history"  # Placeholder for chat history in the prompt template
        )
        # The concrete chat_history content depends on the corresponding user's historical conversation records.
        # Note: the enhanced chain input cannot be a string; it must be a dictionary.
        return conversation_chain

# If the reference examples conflict with the rules above, the reference examples may be given priority.
if __name__ == "__main__":
    # Fixed format: add LangChain configuration and set the session_id for the current program.
    session_config = {
        "configurable": {
            "session_id": "user_test"
        }
    }

    res = RagService().chain.invoke({"input": """
    Please convert the following content to JSON format:
    Monitoring sequence: MDA-MB-231 adherent cells in good condition and in logarithmic growth phase have been prepared and are cultured in DMEM medium containing 10% FBS; execute subtasks 1-2.
    Subtask 1: move the culture dish containing MDA-MB-231 adherent cells to the work area.
    Subtask 2: use a pipette to aspirate the medium from the culture dish and dispense the aspirated medium into the waste container.
    
    Monitoring sequence: after the liquid in the culture dish is mostly removed, execute subtasks 3-5.
    Subtask 3: use a pipette to aspirate 2 mL PBS and add it to the culture dish.
    Subtask 4: gently shake the culture dish so that PBS covers all cells.
    Subtask 5: use a pipette to slowly aspirate PBS from the culture dish and dispense the aspirated liquid into the waste container.

    """}, session_config)
    print(res)
