 #!/usr/bin/env python3
"""
Prompt Templates Library for Windows Blue Screen Debugging

This module contains all prompt templates used in the automated debugging pipeline.
Each template is designed for specific stages of the analysis process.
"""

import json
from pathlib import Path
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any


class PromptTemplateManager:
    """Manager for all debugging prompt templates."""
    
    def __init__(self, whitelisted_commands_path: str = "user_docs/whitelisted_commands.json"):
        """
        Initialize the prompt template manager.
        """
        self.whitelisted_commands_path = Path(whitelisted_commands_path)
        self._whitelisted_commands = self._load_whitelisted_commands()
        
        # Universal signatures for parsing responses
        self.analysis_start_signature = "===== ANALYSIS_START ====="
        self.analysis_end_signature = "===== ANALYSIS_END ====="
        self.command_start_signature = "===== COMMAND_START ====="
        self.command_end_signature = "===== COMMAND_END ====="
        self.stop_command = "STOP_DEBUGGING"
        
        # Final analysis signatures
        self.citations_start_signature = "===== CITATIONS_START ====="
        self.citations_end_signature = "===== CITATIONS_END ====="
        self.final_summary_start_signature = "===== FINAL_SUMMARY_START ====="
        self.final_summary_end_signature = "===== FINAL_SUMMARY_END ====="
    
    def _load_whitelisted_commands(self) -> List[Dict[str, Any]]:
        """Load whitelisted commands from JSON file."""
        try:
            with open(self.whitelisted_commands_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Whitelisted commands file not found: {self.whitelisted_commands_path}") from exc
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in whitelisted commands file: {e}") from e
    
    def _format_whitelisted_commands(self, exclude_commands: List[str] = None) -> str:
        """
        Format whitelisted commands for inclusion in prompts.
        
        Args:
            exclude_commands: List of commands to exclude (previously used commands)
            
        Returns:
            Formatted string of available commands
        """
        if exclude_commands is None:
            exclude_commands = []
            
        # Normalize exclude list (remove spaces and convert to lowercase)
        exclude_normalized = [cmd.lower().replace(' ', '') for cmd in exclude_commands]
        
        commands_text = []
        counter = 1
        
        for cmd in self._whitelisted_commands:
            # Check if this command should be excluded
            cmd_normalized = cmd['command'].lower().replace(' ', '')
            if cmd_normalized in exclude_normalized:
                continue
                
            cmd_text = f"{counter}. **{cmd['name']}** - `{cmd['command']}`\n"
            cmd_text += f"   Purpose: {cmd['purpose']}\n"
            cmd_text += f"   When to use: {cmd['when_to_use']}\n"
            
            if cmd.get('arguments'):
                cmd_text += "   Arguments:\n"
                for arg in cmd['arguments']:
                    cmd_text += f"     - {arg['name']} ({arg['type']}): {arg['description']}\n"
            
            commands_text.append(cmd_text)
            counter += 1
        
        return "\n".join(commands_text)
    
    def _get_role_and_expertise_section(self) -> str:
        """Get the common role and expertise section for all prompts."""
        return """You are an expert Windows kernel-level engineer specializing in debugging blue screen crashes (BSODs). Your task is to analyze crash dump information and guide a systematic debugging process using WinDbg commands.

## YOUR EXPERTISE:
- Deep knowledge of Windows kernel internals, driver architecture, and system crash analysis
- Expert in memory management, I/O subsystems, interrupt handling, and device driver interactions
- Systematic approach to crash analysis using proven WinDbg debugging techniques"""
    
    def _get_analysis_instructions(self, is_initial: bool = True) -> str:
        """Get analysis instructions section."""
        citation_format = """
**CITATION FORMAT**: Include key findings as citations in this format:
[TYPE_OF_METRIC] : [ACTUAL_VALUE] : short analysis
Examples:
[BUGCHECK_CODE] : 0x4a : IRQL violation indicating driver failed to lower IRQL
[FAULTING_MODULE] : mydriver.sys : Third-party driver with known stability issues
[PROCESS_NAME] : notmyfault64.e : Test application deliberately triggering crashes"""
        
        if is_initial:
            return f"""### ANALYSIS REQUIRED:
Provide comprehensive analysis including:
- **Crash Type & Bugcheck**: What type of crash and what the bugcheck indicates
- **Probable Root Cause**: Your assessment of what likely caused this crash
- **Key Evidence**: Specific evidence that supports your assessment
- **Initial Hypothesis**: Your working theory about what happened
{citation_format}

**Wrap your analysis between these signatures:**
{self.analysis_start_signature}
[Your detailed analysis here]
{self.analysis_end_signature}"""
        else:
            return f"""### CONTINUED ANALYSIS:
Based on retrieved context and previous outputs, provide:
- **New Insights**: Additional information learned from recent command outputs
- **Updated Hypothesis**: How your understanding has evolved
- **Root Cause Assessment**: Are you closer to identifying the definitive cause?
- **Confidence Level**: How confident are you (Low/Medium/High)?
{citation_format}

**Wrap your analysis between these signatures:**
{self.analysis_start_signature}
[Your detailed analysis here]
{self.analysis_end_signature}"""
    
    def _get_command_instructions(self, is_initial: bool = True) -> str:
        """Get command selection instructions."""
        stop_option = "" if is_initial else f"""
**STOP Option**: If you are 100% confident you've found the definitive root cause with strong evidence, respond with `{self.stop_command}` instead of a WinDbg command."""
        
        return f"""### COMMAND SELECTION:
Choose exactly ONE command from the available list below.
- Extract specific values (addresses, etc.) from the crash data - no placeholders
- Provide the complete, executable command{stop_option}

**Wrap your command between these signatures:**
{self.command_start_signature}
[Your exact command here]
{self.command_end_signature}"""
    
    def get_initial_prompt_template(self) -> PromptTemplate:
        """
        Get the initial prompt template for starting the debugging pipeline.
        Returns: PromptTemplate configured for initial analysis
        """
        template = f"""{self._get_role_and_expertise_section()}

## CRASH DUMP DATA:
```
{{analyze_v_output}}
```

{self._get_analysis_instructions(is_initial=True)}

{self._get_command_instructions(is_initial=True)}

## AVAILABLE COMMANDS:
{self._format_whitelisted_commands()}

Begin your analysis."""

        return PromptTemplate(
            input_variables=["analyze_v_output"],
            template=template
        )
    
    def get_main_debugging_prompt_template(self) -> PromptTemplate:
        """
        Get the main debugging pipeline prompt template for continued analysis.
        Returns: PromptTemplate configured for main debugging pipeline
        """
        template = f"""{self._get_role_and_expertise_section()}

## DEBUGGING CONTEXT:
### Retrieved Context:
```
{{rag_context}}
```

### Previously Executed:
{{previous_commands}}

{self._get_analysis_instructions(is_initial=False)}

{self._get_command_instructions(is_initial=False)}

## AVAILABLE COMMANDS:
{{available_commands}}

Continue your analysis."""

        return PromptTemplate(
            input_variables=["rag_context", "previous_commands", "available_commands"],
            template=template
        )
    
    def get_final_analysis_prompt_template(self) -> PromptTemplate:
        """Get the final comprehensive analysis prompt template."""
        template = f"""{self._get_role_and_expertise_section()}

## DEBUGGING COMPLETE - FINAL ANALYSIS
The debugging process has concluded. You have identified the definitive root cause of this crash. Below is the complete context of all your previous analysis and findings from the debugging session:

## COMPLETE DEBUGGING CONTEXT:
```
{{complete_analysis_context}}
```

## FINAL REPORT REQUIRED:
Create a comprehensive final report with two sections:

### 1. CITATIONS
List all concrete metrics and evidence that led to your conclusion. Each citation should follow this format:
[TYPE_OF_METRIC] : [ACTUAL_VALUE] : 1-2 sentence description

Include citations for key findings such as:
- BUGCHECK_CODE, FAULTING_MODULE, PROCESS_NAME, THREAD_ADDRESS
- IRQL_LEVEL, STACK_FRAME, MEMORY_ADDRESS, DRIVER_VERSION
- Any other concrete evidence that supports your conclusion

**Wrap citations between these signatures:**
{self.citations_start_signature}
[Your citations here]
{self.citations_end_signature}

### 2. FINAL SUMMARY
Provide a human-readable summary (4-5 sentences) that:
- Explains what the issue is in simple terms
- Describes the impact or risk level
- Suggests any immediate actions that can be taken
- Keeps technical jargon to a minimum

**Wrap summary between these signatures:**
{self.final_summary_start_signature}
[Your summary here]
{self.final_summary_end_signature}

Begin your final analysis."""

        return PromptTemplate(
            input_variables=["complete_analysis_context"],
            template=template
        )
    
    def extract_analysis_from_response(self, response: str) -> str:
        """
        Extract analysis section from LLM response (works for both initial and main templates).
        """
        try:
            start_idx = response.find(self.analysis_start_signature)
            end_idx = response.find(self.analysis_end_signature)
            
            if start_idx == -1 or end_idx == -1:
                return ""
            
            start_idx += len(self.analysis_start_signature)
            analysis = response[start_idx:end_idx].strip()
            return analysis
            
        except (ValueError, AttributeError):
            return ""
    
    def extract_command_from_response(self, response: str) -> str:
        """
        Extract command from LLM response (works for both initial and main templates).
        """
        try:
            start_idx = response.find(self.command_start_signature)
            end_idx = response.find(self.command_end_signature)
            
            if start_idx == -1 or end_idx == -1:
                return ""
            
            start_idx += len(self.command_start_signature)
            command = response[start_idx:end_idx].strip()
            return command
            
        except (ValueError, AttributeError):
            return ""
    
    def is_stop_command(self, command: str) -> bool:
        return command.strip() == self.stop_command
    

    
    def format_previous_commands(self, commands: List[str]) -> str:
        if not commands:
            return "None - This is the first command in the debugging session."
        
        formatted = []
        for i, cmd in enumerate(commands, 1):
            formatted.append(f"{i}. {cmd}")
        
        return "\n".join(formatted)
    
    def get_main_prompt_with_context(self, 
                                   rag_context: str, 
                                   previous_commands: List[str]) -> str:
        """
        Get a formatted main debugging prompt with context and filtered commands.
        """
        template = self.get_main_debugging_prompt_template()
        
        # Format previous commands
        formatted_previous = self.format_previous_commands(previous_commands)
        
        # Get available commands (excluding previously used ones)
        available_commands = self._format_whitelisted_commands(exclude_commands=previous_commands)
        
        return template.format(
            rag_context=rag_context,
            previous_commands=formatted_previous,
            available_commands=available_commands
        )
    
    def process_llm_response(self, response: str) -> dict:
        """Process LLM response and extract all relevant information."""
        analysis = self.extract_analysis_from_response(response)
        command = self.extract_command_from_response(response)
        is_stop = self.is_stop_command(command) if command else False
        
        return {
            "analysis": analysis,
            "command": command if not is_stop else None,
            "is_stop": is_stop
        }
    

