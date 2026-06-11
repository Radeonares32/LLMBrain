# Brainframe

## What is a Brainframe?

A **Brainframe** is the internal semantic representation format used by LLM Brain to unify disparate engineering artifacts into a single cohesive structure.

## Purpose

Different engineering artifacts have different structures:
- Source code is hierarchical and syntactic.
- ADRs are narrative and decision-focused.
- Incidents are chronological and symptom-focused.

The Brainframe normalizes these into a common schema that an LLM can easily reason over without needing specialized prompt engineering for every single data type.

## Schema Structure

A typical Brainframe object consists of:
- `id`: Unique identifier across the entire memory.
- `type`: Code, Doc, ADR, Incident, SecurityNote.
- `content`: The normalized text content.
- `metadata`: Source file, authorship, timestamp, tags.
- `relations`: Edges to other Brainframes (e.g., `IMPLEMENTS_ADR`, `RESOLVES_INCIDENT`).

## Usage

When an LLM queries the LLM Brain, the response is typically formatted as a collection of relevant Brainframes, allowing the model to see not just the text, but how the text relates to the broader engineering context.
