# Hiram - Cyber Security Architecture Tool

This project is a tool called Hiram that aims to assist cyber security architects and advisors in designing secure systems. It leverages machine learning and fuzzy matching techniques to find the best-matching pre-approved architecture based on a suggested design.

## Introduction

During the integration and design process, a cyber security architect/consultant often asks developers or IT administrators to draw the proposed architecture. However, reviewing and suggesting improvements based on security best practices can be time-consuming and costly. Hiram solves this problem by using machine learning to match the suggested design with pre-approved architectures, providing instant feedback to developers and saving time for the cyber security personnel.

## Methodology

Hiram follows the following methodology:
1. The data is received as an XML file saved from DRAW.IO.
2. The XML file is decompressed using the Deflate compression algorithm.
3. The decompressed XML is converted into a 2-dimensional matrix representing the connections between different components.
4. The data is normalized into a format suitable for fuzzy learning using machine learning algorithms.
5. The 2-dimensional matrix is transformed into a single binary word.
6. The Levenshtein Distance algorithm is used to find the best-known word that matches the new word.

## Possible Improvements

To further categorize the new architectures, a grouping algorithm can be applied. Each component can be divided into a general category, and the number of components in each category can be counted. By placing the resulting vectors on an N-dimensional graph, the nearest distance to a pre-categorized architecture can be chosen for fuzzy matching.

## Scope

The scope of this work includes seven different items described in the XML's shape attribute:
1. Thin client item
2. Firewall
3. Xenapp server (represents a file server)
4. Cache server (represents a web server)
5. Cellphone
6. Xenclient synchronizer (represents a back-end server)
7. Chassis (represents an SQL server)
8. Direction line (indicates the source-to-destination packet flow)

Mixing and matching about 12 different architectures from different domains, such as web applications, backup applications, mobile applications, and logging servers, will provide a learning base.

## Goals

The project aims to receive any architecture from Draw.io containing the specified objects and offer a more secure alternative while preserving the original design's intention. For example, if a web application architecture is detected, the offered architecture should be able to perform the same operations but with a higher level of security. The reliability of the tool will be tested with three architectures from each domain.

## Related Work

In the related work section, we review previous research related to our study. We provide an overview of literature on automatic architecture review and summarize relevant studies.

### Literature on Automatic Architecture Review

1. "Nemesis: Automated Architecture for Threat Modeling and Risk Assessment for Cloud Computing" by Patrick Kamongi, Mahadevan Gomathisankaran, and Krishna Kavi (2014): This paper discusses checking a cloud configuration for vulnerabilities by enumerating the different components and evaluating their vulnerabilities. However, it does not address the issue of correct architecture eliminating many risks.

2. "ARES: Automated Risk Estimation in Smart Sensor Environments" by Athanasios Dimitriadis (July 2020): This paper presents a model for risk assessment but focuses on identifying problems with a specific architecture drawn in a non-well-known industry tool, rather than providing a better architecture.

3. "A Fuzzy Probability Bayesian Network Approach for Dynamic Cybersecurity Risk Assessment in Industrial Control Systems" by Qi Zhang and Ch
