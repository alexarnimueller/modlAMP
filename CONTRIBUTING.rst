Contribution Guide To modlAMP
=============================

**Everyone is welcome to contribute!**

If you plan to contribute to this project, please comply with the following guidelines:


Contribution Guidelines
-----------------------

Before starting, get a sense of which functionality and modules are already included in **modlAMP** and read the
documentation. If you then still think, your idea is missing, go for it!

- Comply with the used docstring format (rst, check the existing code and comment in the same way) --> documentation is automatically generated
- If you extend an existing module/class/function, use the same variable names.
- For all of your new modules/classes/functions, make Unittest test cases, otherwise nobody knows, whether your feature actually works!
- Include code examples of how to use your amazing new feature / add a section to the README.rst file


Git Workflow
............

1) clone a version of the current Git repository to your local machine
2) create a new git branch for your project: ``git branch <branch-name>``
3) change to your new branch: ``git checkout <branch-name>``
4) start implementing your amazing new feature!
5) regularly ``git add`` and ``git commit``
6) if you feel like your amazing new feature is ready to be used by others, push your branch back to the origin: ``git push origin <branch-name>``
7) if your finished implementing the feature, create a **merge request** on the GitLab server to include your branch into the master branch.