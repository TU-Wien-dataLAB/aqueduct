---
title: Teams
parent: User Guide
nav_order: 2
---

# Teams
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Teams View

The Teams page displays a list of teams within your organization.

![Teams Page](../assets/user_guide/teams_page.png)

You can view the default organization limits, which apply to tokens when no team or user-specific limits are set.
Limits include Requests per Minute, Input Tokens per Minute, and Output Tokens per Minute. If you are an admin, you can modify these
limits in the Admin Panel.

**If you exceed any of these limits when making a request, the API will respond with a 429 HTTP error (Too Many Requests).**

As an Organization Admin, you can create new teams using the "Create New Team" button, which opens the team creation form. Here,
you can set the team name and description.

You can also delete teams using the delete ("🗑") button in the team list if you have admin privileges.

## Team Detail View

Clicking on a team in the list opens the Team Detail page.

![Team Detail Page](../assets/user_guide/team_detail.png)

On this page, you can view the team limits, which function similarly to organization limits. If no team limit is set, the
organization limit is used by default.

Below the limits, you’ll see the service accounts associated with the team. If you have Team Admin privileges, you can create
service accounts using the "Add Service Account" button. Service accounts require a name and description, and a token is
generated upon creation. If you create a service account, you own the associated token.

You can transfer ownership of a service account token to another user within the team. If you own a service
account, you cannot be removed from the team.

You can regenerate service account tokens, and you can delete service accounts, which also deletes the associated token.
You can also edit service accounts to change their name and description.

Below the service accounts list, you’ll see the team members. Each member has a tag indicating their permissions (User,
Team-Admin, Org-Admin, or Admin). If you are a Team-Admin, you can promote standard users to Team-Admin. If you do not have any service accounts, you can be removed from the team.

If a user is not part of a team, they can be added using the "Add Users to Team" function. Any user in the organization can
be added to a team.
