---
title: Models
parent: User Guide
nav_order: 4
---

# Models

On the Models page, you can view the available models.

![Models Page](../assets/user_guide/models_page.png)

The model name in the table specifies the name for the model to use when making requests.
The TPM and RPM columns specify the tokens per minute and requests per minute for this model.
The litellm_params show additional information like the model timeout in seconds.

## Model Aliases

Model aliases are alternative names for models. When making API requests, you can use either the model name or any of its aliases in the `model` parameter. Aliases are configured in your router configuration file. 