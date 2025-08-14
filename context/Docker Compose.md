========================
CODE SNIPPETS
========================
TITLE: Docker Compose Start Command API Reference
DESCRIPTION: Detailed API documentation for the `docker compose start` command, including its purpose and all available command-line options with their types and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_start.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose start
Description: Starts existing containers for a service

Options:
  --dry-run:
    Type: bool
    Description: Execute command in dry run mode
```

----------------------------------------

TITLE: Example Usage of docker compose top
DESCRIPTION: Demonstrates how to use the `docker compose top` command and its typical output, showing process details for a service named 'example_foo_1'.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_top.md#_snippet_1

LANGUAGE: console
CODE:
```
$ docker compose top
example_foo_1
UID    PID      PPID     C    STIME   TTY   TIME       CMD
root   142353   142331   2    15:33   ?     00:00:00   ping localhost -c 5
```

----------------------------------------

TITLE: Start Docker Compose Application
DESCRIPTION: This command initiates and runs the entire application defined in the `compose.yaml` file. It will build images if necessary, create containers, and start all defined services in an isolated environment.

SOURCE: https://github.com/docker/compose/blob/main/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
docker compose up
```

----------------------------------------

TITLE: Example: Previewing Docker Compose `up` with Dry Run
DESCRIPTION: This console output demonstrates the use of `docker compose --dry-run up --build -d`. It shows the sequence of operations Compose would perform, including pulling images, building services, creating containers, and starting them, all without altering the actual stack state. The output clearly indicates each step is a 'DRY-RUN MODE' action, illustrating the order of operations from image pulling and service building to container creation and health checks.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_11

LANGUAGE: console
CODE:
```
$ docker compose --dry-run up --build -d
[+] Pulling 1/1
 ✔ DRY-RUN MODE -  db Pulled                                                                                                                                                                                                               0.9s
[+] Running 10/8
 ✔ DRY-RUN MODE -    build service backend                                                                                                                                                                                                 0.0s
 ✔ DRY-RUN MODE -  ==> ==> writing image dryRun-754a08ddf8bcb1cf22f310f09206dd783d42f7dd                                                                                                                                                   0.0s
 ✔ DRY-RUN MODE -  ==> ==> naming to nginx-golang-mysql-backend                                                                                                                                                                            0.0s
 ✔ DRY-RUN MODE -  Network nginx-golang-mysql_default                                    Created                                                                                                                                           0.0s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-db-1                                     Created                                                                                                                                           0.0s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-backend-1                                Created                                                                                                                                           0.0s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-proxy-1                                  Created                                                                                                                                           0.0s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-db-1                                     Healthy                                                                                                                                           0.5s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-backend-1                                Started                                                                                                                                           0.0s
 ✔ DRY-RUN MODE -  Container nginx-golang-mysql-proxy-1                                  Started                                     Started
```

----------------------------------------

TITLE: Example Docker Compose Configuration for Services
DESCRIPTION: A sample `compose.yaml` file demonstrating the definition of two services: `db` using the `postgres` image and `web` with a build context, volume mounts, port mappings, and a dependency on the `db` service.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_pull.md#_snippet_1

LANGUAGE: yaml
CODE:
```
services:
  db:
    image: postgres
  web:
    build: .
    command: bundle exec rails s -p 3000 -b '0.0.0.0'
    volumes:
      - .:/myapp
    ports:
      - "3000:3000"
    depends_on:
      - db
```

----------------------------------------

TITLE: Build Docker Compose CLI Plugin
DESCRIPTION: This command compiles the Docker Compose CLI plugin for the host machine. The resulting binary will be located in the `./bin/build` directory.

SOURCE: https://github.com/docker/compose/blob/main/BUILDING.md#_snippet_0

LANGUAGE: console
CODE:
```
make
```

----------------------------------------

TITLE: Combine Multiple Docker Compose Configuration Files
DESCRIPTION: This example demonstrates how to use the `-f` flag to specify and combine multiple Compose configuration files. Compose merges these files in the order provided, allowing subsequent files to override or add to previous configurations, enabling modular project setups.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_3

LANGUAGE: console
CODE:
```
$ docker compose -f compose.yaml -f compose.admin.yaml run backup_db
```

----------------------------------------

TITLE: Base `compose.yaml` Service Definition Example
DESCRIPTION: This YAML snippet illustrates a basic `compose.yaml` file defining a `webapp` service. It includes common configurations like image, port mapping, and volume, serving as the foundational configuration when multiple Compose files are layered.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_4

LANGUAGE: yaml
CODE:
```
services:
  webapp:
    image: examples/web
    ports:
      - "8000:8000"
    volumes:
      - "/data"
```

----------------------------------------

TITLE: Run Docker Compose CLI Unit Tests
DESCRIPTION: Commands to execute all unit tests for the Docker Compose CLI. An additional command is provided for updating golden files, which is useful during Go test development.

SOURCE: https://github.com/docker/compose/blob/main/BUILDING.md#_snippet_1

LANGUAGE: console
CODE:
```
make test
```

LANGUAGE: console
CODE:
```
go test ./... -test.update-golden
```

----------------------------------------

TITLE: Execute All Docker Compose CLI End-to-End Tests
DESCRIPTION: Commands to run the entire suite of end-to-end tests, encompassing both CLI and standalone components. Options include running tests directly or building the CLI first before execution.

SOURCE: https://github.com/docker/compose/blob/main/BUILDING.md#_snippet_2

LANGUAGE: console
CODE:
```
make e2e
```

LANGUAGE: console
CODE:
```
make build-and-e2e
```

----------------------------------------

TITLE: Activate Docker Compose Profiles for Optional Services
DESCRIPTION: Illustrates how to use the `--profile` flag to activate specific service profiles when bringing up Docker Compose services. Examples include activating a single profile (`frontend`) and activating multiple profiles (`frontend` and `debug`) simultaneously.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_8

LANGUAGE: console
CODE:
```
docker compose --profile frontend up
docker compose --profile frontend --profile debug up
```

----------------------------------------

TITLE: Execute Docker Compose CLI Plugin End-to-End Tests
DESCRIPTION: Commands to run end-to-end tests specifically for the Docker Compose CLI plugin. This includes options to either run the tests directly or to build the CLI beforehand.

SOURCE: https://github.com/docker/compose/blob/main/BUILDING.md#_snippet_3

LANGUAGE: console
CODE:
```
make e2e-compose
```

LANGUAGE: console
CODE:
```
make build-and-e2e-compose
```

----------------------------------------

TITLE: Execute Docker Compose CLI Standalone End-to-End Tests
DESCRIPTION: Commands to run end-to-end tests for the standalone Docker Compose CLI. Options are provided to either execute the tests directly or to build the CLI prior to running the tests.

SOURCE: https://github.com/docker/compose/blob/main/BUILDING.md#_snippet_4

LANGUAGE: console
CODE:
```
make e2e-compose-standalone
```

LANGUAGE: console
CODE:
```
make build-and-e2e-compose-standalone
```

----------------------------------------

TITLE: Docker Compose Service Configuration for Image Push
DESCRIPTION: Example `docker-compose.yml` configuration demonstrating how to define services with images destined for a local registry or Docker Hub, which can then be pushed using `docker compose push`.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_push.md#_snippet_0

LANGUAGE: yaml
CODE:
```
services:
  service1:
    build: .
    image: localhost:5000/yourimage  ## goes to local registry

  service2:
    build: .
    image: your-dockerid/yourimage  ## goes to your repository on Docker Hub
```

----------------------------------------

TITLE: Run Docker Compose Service Without Dependencies
DESCRIPTION: This example uses the `--no-deps` flag to execute a command within the `web` service without starting any linked containers. This is beneficial when you only need to run a command on a specific service in isolation, without bringing up its entire dependency tree. It helps in faster execution for isolated tasks.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_11

LANGUAGE: console
CODE:
```
docker compose run --no-deps web python manage.py shell
```

----------------------------------------

TITLE: Declaring Service Dependency on an External Provider
DESCRIPTION: This YAML snippet demonstrates how an application service (`app`) can declare a dependency on another service (`database`) that is managed by an external provider. The `depends_on` attribute ensures that the `database` service is started before `app`. This setup allows Compose to inject environment variables from the provider into the dependent service.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_4

LANGUAGE: yaml
CODE:
```
services:
  app:
    image: myapp 
    depends_on:
       - database

  database:
    provider:
      type: awesomecloud
```

----------------------------------------

TITLE: Kill Docker Compose Containers with Custom Signal
DESCRIPTION: This example demonstrates how to use the `docker compose kill` command to send a `SIGINT` signal to containers instead of the default `SIGKILL`.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_kill.md#_snippet_0

LANGUAGE: console
CODE:
```
$ docker compose kill -s SIGINT
```

----------------------------------------

TITLE: Adding a Signed-off-by Line to Git Commits
DESCRIPTION: An example of the 'Signed-off-by' line, which must be appended to every Git commit message. This line serves as a formal certification that the contributor agrees to the terms of the Developer Certificate of Origin (DCO).

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_4

LANGUAGE: git
CODE:
```
Signed-off-by: Joe Smith <joe.smith@email.com>
```

----------------------------------------

TITLE: Control Docker Compose Parallelism with --parallel Flag
DESCRIPTION: Demonstrates how to limit the maximum level of concurrent engine calls using the `--parallel` flag. The example shows pulling images one at a time by setting parallelism to 1.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_9

LANGUAGE: console
CODE:
```
docker compose --parallel 1 pull
```

----------------------------------------

TITLE: Specify Docker Compose Project Name with -p Flag
DESCRIPTION: Demonstrates how to use the `-p` flag to set a custom project name for Docker Compose commands. The example shows `ps -a` to list services and `logs` to view service logs, both operating under the specified project name `my_project`.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_7

LANGUAGE: console
CODE:
```
$ docker compose -p my_project ps -a
NAME                 SERVICE    STATUS     PORTS
my_project_demo_1    demo       running

$ docker compose -p my_project logs
demo_1  | PING localhost (127.0.0.1): 56 data bytes
demo_1  | 64 bytes from 127.0.0.1: seq=0 ttl=64 time=0.095 ms
```

----------------------------------------

TITLE: Run Bash in Docker Compose Web Service
DESCRIPTION: This command starts the `web` service and executes `bash` inside a new container. It demonstrates the basic usage of `docker compose run` to override a service's default command, providing an interactive shell within the service's environment.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_0

LANGUAGE: console
CODE:
```
$ docker compose run web bash
```

----------------------------------------

TITLE: Run PostgreSQL Shell for Linked Service in Docker Compose
DESCRIPTION: This command opens an interactive PostgreSQL shell (`psql`) within a new container based on the `db` service. It automatically checks if the linked `db` service is running and starts it if necessary, demonstrating how `docker compose run` handles dependencies.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_3

LANGUAGE: console
CODE:
```
$ docker compose run db psql -h db -U docker
```

----------------------------------------

TITLE: Specify Custom Path for a Single Docker Compose File
DESCRIPTION: This example demonstrates using the `-f` flag to specify a Compose file located at an arbitrary path, not necessarily in the current directory. This is useful for managing projects from a centralized location or when the Compose file resides elsewhere on the filesystem.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_6

LANGUAGE: console
CODE:
```
$ docker compose -f ~/sandbox/rails/compose.yaml pull db
```

----------------------------------------

TITLE: Run Command Without Dependencies in Docker Compose
DESCRIPTION: This command executes `python manage.py shell` in the `web` service container without starting any linked services. The `--no-deps` flag prevents `docker compose run` from automatically bringing up dependent containers, useful for isolated execution.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_4

LANGUAGE: console
CODE:
```
$ docker compose run --no-deps web python manage.py shell
```

----------------------------------------

TITLE: Run Docker Compose Service with Exposed Ports
DESCRIPTION: This example illustrates using the `--service-ports` flag with `docker compose run` to expose ports defined in the service configuration. By default, `run` does not publish ports to avoid collisions, but this flag ensures the service's configured ports are mapped to the host. It's useful when the command needs network access via the service's defined ports.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_8

LANGUAGE: console
CODE:
```
docker compose run --service-ports web python manage.py shell
```

----------------------------------------

TITLE: Example JSON Output for Docker Compose Events
DESCRIPTION: This snippet illustrates the structure of a single JSON object emitted by the `docker compose events` command when the `--json` flag is enabled. It shows typical fields such as timestamp, event type, action, container ID, service name, and additional attributes like container name and image.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_events.md#_snippet_0

LANGUAGE: json
CODE:
```
{
    "time": "2015-11-20T18:01:03.615550",
    "type": "container",
    "action": "create",
    "id": "213cf7...5fc39a",
    "service": "web",
    "attributes": {
      "name": "application_web_1",
      "image": "alpine:edge"
    }
}
```

----------------------------------------

TITLE: Run Interactive PostgreSQL Shell in Docker Compose DB Service
DESCRIPTION: This snippet demonstrates connecting to an interactive PostgreSQL shell within the `db` service. The `docker compose run` command automatically checks and starts any linked services (like `db` in this case) before executing the specified command. This simplifies interacting with dependent services for administrative tasks.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_10

LANGUAGE: console
CODE:
```
docker compose run db psql -h db -U docker
```

----------------------------------------

TITLE: docker compose build Command-Line Options
DESCRIPTION: Reference for the command-line arguments and flags available when executing `docker compose build`, including their data types, default values, and a brief explanation of their purpose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_build.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
--build-arg:
  Type: stringArray
  Default: 
  Description: Set build-time variables for services
--builder:
  Type: string
  Default: 
  Description: Set builder to use
--check:
  Type: bool
  Default: 
  Description: Check build configuration
--dry-run:
  Type: bool
  Default: 
  Description: Execute command in dry run mode
-m, --memory:
  Type: bytes
  Default: 0
  Description: Set memory limit for the build container. Not supported by BuildKit.
--no-cache:
  Type: bool
  Default: 
  Description: Do not use cache when building the image
--print:
  Type: bool
  Default: 
  Description: Print equivalent bake file
--pull:
  Type: bool
  Default: 
  Description: Always attempt to pull a newer version of the image
--push:
  Type: bool
  Default: 
  Description: Push service images
-q, --quiet:
  Type: bool
  Default: 
  Description: Don't print anything to STDOUT
--ssh:
  Type: string
  Default: 
  Description: Set SSH authentications used when building service images. (use 'default' for using your default SSH Agent)
--with-dependencies:
  Type: bool
  Default: 
  Description: Also build dependencies (transitively)
```

----------------------------------------

TITLE: Docker Compose Version Command Options
DESCRIPTION: Documents the available options for the `docker compose version` command, including dry-run, output format, and short display options, along with their types and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_version.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose version:
  Options:
    --dry-run:
      Type: bool
      Default: 
      Description: Execute command in dry run mode
    -f, --format:
      Type: string
      Default: 
      Description: Format the output. Values: [pretty | json]. (Default: pretty)
    --short:
      Type: bool
      Default: 
      Description: Shows only Compose's version number
```

----------------------------------------

TITLE: Docker Compose Create Command Options
DESCRIPTION: This section details the various command-line options available for the `docker compose create` command. Each option includes its name, data type, default value (if applicable), and a brief description of its purpose and effect on the command's execution.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_create.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose create options:
  --build: bool - Build images before starting containers
  --dry-run: bool - Execute command in dry run mode
  --force-recreate: bool - Recreate containers even if their configuration and image haven't changed
  --no-build: bool - Don't build an image, even if it's policy
  --no-recreate: bool - If containers already exist, don't recreate them. Incompatible with --force-recreate.
  --pull: string (policy) - Pull image before running ("always"|"missing"|"never"|"build")
  --quiet-pull: bool - Pull without printing progress information
  --remove-orphans: bool - Remove containers for services not defined in the Compose file
  --scale: stringArray - Scale SERVICE to NUM instances. Overrides the `scale` setting in the Compose file if present.
  -y, --yes: bool - Assume "yes" as answer to all prompts and run non-interactively
```

----------------------------------------

TITLE: Docker Compose Alpha CLI Reference
DESCRIPTION: Reference documentation for the experimental `docker compose alpha` commands, detailing available subcommands like `viz` and `watch`, and global options such as `--dry-run`.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose alpha:
  Description: Experimental commands

  Subcommands:
    viz:
      Description: EXPERIMENTAL - Generate a graphviz graph from your compose file
      Reference: compose_alpha_viz.md
    watch:
      Description: EXPERIMENTAL - Watch build context for service and rebuild/refresh containers when files are updated
      Reference: compose_alpha_watch.md

  Options:
    --dry-run:
      Type: (flag)
      Default: (none)
      Description: Execute command in dry run mode
```

----------------------------------------

TITLE: docker compose alpha generate Command API Reference
DESCRIPTION: API documentation for the `docker compose alpha generate` command, detailing its purpose, experimental status, and available command-line options with their types, defaults, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha_generate.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose alpha generate
Description: EXPERIMENTAL - Generate a Compose file from existing containers

Options:
  --dry-run:
    Type: bool
    Default: (none)
    Description: Execute command in dry run mode
  --format:
    Type: string
    Default: yaml
    Description: Format the output. Values: [yaml | json]
  --name:
    Type: string
    Default: (none)
    Description: Project name to set in the Compose file
  --project-dir:
    Type: string
    Default: (none)
    Description: Directory to use for the project
```

----------------------------------------

TITLE: Docker Compose Command Options Reference
DESCRIPTION: Detailed reference for command-line options available when running Docker Compose commands, including their types, default values, and descriptions. This covers options for managing container lifecycle, build processes, logging, and dependency handling.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_up.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Docker Compose Command Options:
  - Name: --abort-on-container-exit
    Type: bool
    Default: N/A
    Description: Stops all containers if any container was stopped. Incompatible with -d
  - Name: --abort-on-container-failure
    Type: bool
    Default: N/A
    Description: Stops all containers if any container exited with failure. Incompatible with -d
  - Name: --always-recreate-deps
    Type: bool
    Default: N/A
    Description: Recreate dependent containers. Incompatible with --no-recreate.
  - Name: --attach
    Type: stringArray
    Default: N/A
    Description: Restrict attaching to the specified services. Incompatible with --attach-dependencies.
  - Name: --attach-dependencies
    Type: bool
    Default: N/A
    Description: Automatically attach to log output of dependent services
  - Name: --build
    Type: bool
    Default: N/A
    Description: Build images before starting containers
  - Name: -d, --detach
    Type: bool
    Default: N/A
    Description: Detached mode: Run containers in the background
  - Name: --dry-run
    Type: bool
    Default: N/A
    Description: Execute command in dry run mode
  - Name: --exit-code-from
    Type: string
    Default: N/A
    Description: Return the exit code of the selected service container. Implies --abort-on-container-exit
  - Name: --force-recreate
    Type: bool
    Default: N/A
    Description: Recreate containers even if their configuration and image haven't changed
  - Name: --menu
    Type: bool
    Default: N/A
    Description: Enable interactive shortcuts when running attached. Incompatible with --detach. Can also be enable/disable by setting COMPOSE_MENU environment var.
  - Name: --no-attach
    Type: stringArray
    Default: N/A
    Description: Do not attach (stream logs) to the specified services
  - Name: --no-build
    Type: bool
    Default: N/A
    Description: Don't build an image, even if it's policy
  - Name: --no-color
    Type: bool
    Default: N/A
    Description: Produce monochrome output
  - Name: --no-deps
    Type: bool
    Default: N/A
    Description: Don't start linked services
  - Name: --no-log-prefix
    Type: bool
    Default: N/A
    Description: Don't print prefix in logs
  - Name: --no-recreate
    Type: bool
    Default: N/A
    Description: If containers already exist, don't recreate them. Incompatible with --force-recreate.
  - Name: --no-start
    Type: bool
    Default: N/A
    Description: Don't start the services after creating them
  - Name: --pull
    Type: string
    Default: policy
    Description: Pull image before running ("always"|"missing"|"never")
  - Name: --quiet-pull
    Type: bool
    Default: N/A
    Description: Pull without printing progress information
  - Name: --remove-orphans
    Type: bool
    Default: N/A
    Description: Remove containers for services not defined in the Compose file
```

----------------------------------------

TITLE: Docker Compose Scale Command API Reference
DESCRIPTION: This section provides an API-like reference for the `docker compose scale` command, listing its primary function and all supported command-line options with their types and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_scale.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose scale
Description: Scale services

Options:
  --dry-run (Type: bool, Default: None) - Execute command in dry run mode
  --no-deps (Type: bool, Default: None) - Don't start linked services
```

----------------------------------------

TITLE: Docker Compose Images Command Options
DESCRIPTION: Documents the available command-line options for the `docker compose images` command, including their types, default values, and descriptions. This section helps users understand how to customize the command's behavior and output.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_images.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose images
Description: List images used by the created containers

Options:
  --dry-run (bool): Execute command in dry run mode
  --format (string, default: table): Format the output. Values: [table | json]
  -q, --quiet (bool): Only display IDs
```

----------------------------------------

TITLE: Docker Compose `ps` Command Options Reference
DESCRIPTION: Reference documentation for command-line options available with `docker compose ps`, detailing their types and purpose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_3

LANGUAGE: APIDOC
CODE:
```
Option: --status
  Type: stringArray
  Description: Filter services by status.
  Allowed Values: paused, restarting, removing, running, dead, created, exited
```

----------------------------------------

TITLE: Docker Compose Up Command Behavior and Usage
DESCRIPTION: Explains the core functionality of `docker compose up`, including how it handles service creation, linking, output aggregation, background execution, configuration changes, and exit codes.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_up.md#_snippet_2

LANGUAGE: APIDOC
CODE:
```
Builds, (re)creates, starts, and attaches to containers for a service.

Unless they are already running, this command also starts any linked services.

The `docker compose up` command aggregates the output of each container (like `docker compose logs --follow` does).
One can optionally select a subset of services to attach to using `--attach` flag, or exclude some services using 
`--no-attach` to prevent output to be flooded by some verbose services. 

When the command exits, all containers are stopped. Running `docker compose up --detach` starts the containers in the
background and leaves them running.

If there are existing containers for a service, and the service’s configuration or image was changed after the
container’s creation, `docker compose up` picks up the changes by stopping and recreating the containers
(preserving mounted volumes). To prevent Compose from picking up changes, use the `--no-recreate` flag.

If you want to force Compose to stop and recreate all containers, use the `--force-recreate` flag.

If the process encounters an error, the exit code for this command is `1`.
If the process is interrupted using `SIGINT` (ctrl + C) or `SIGTERM`, the containers are stopped, and the exit code is `0`.
```

----------------------------------------

TITLE: docker compose wait Command Options
DESCRIPTION: Details the command-line options available for the `docker compose wait` command, including their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_wait.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Options:
  --down-project:
    Type: bool
    Default: 
    Description: Drops project when the first container stops
  --dry-run:
    Type: bool
    Default: 
    Description: Execute command in dry run mode
```

----------------------------------------

TITLE: Docker Compose Up Command Options
DESCRIPTION: Lists the command-line options available for `docker compose up`, specifying their type, default value, and a brief explanation of their function.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_up.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
| `-V`, `--renew-anon-volumes`   | `bool`        |          | Recreate anonymous volumes instead of retrieving data from the previous containers                                                                  |
| `--scale`                      | `stringArray` |          | Scale SERVICE to NUM instances. Overrides the `scale` setting in the Compose file if present.                                                       |
| `-t`, `--timeout`              | `int`         | `0`      | Use this timeout in seconds for container shutdown when attached or when containers are already running                                             |
| `--timestamps`                 | `bool`        |          | Show timestamps                                                                                                                                     |
| `--wait`                       | `bool`        |          | Wait for services to be running\|healthy. Implies detached mode.                                                                                    |
| `--wait-timeout`               | `int`         | `0`      | Maximum duration in seconds to wait for the project to be running\|healthy                                                                          |
| `-w`, `--watch`                | `bool`        |          | Watch source code and rebuild/refresh containers when files are updated.                                                                            |
| `-y`, `--yes`                  | `bool`        |          | Assume "yes" as answer to all prompts and run non-interactively                                                                                     |
```

----------------------------------------

TITLE: docker compose alpha publish Command Options
DESCRIPTION: Detailed options available for the 'docker compose alpha publish' command, including their types, default values, and descriptions for controlling the publishing process.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha_publish.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose alpha publish

Options:
  --dry-run:
    Type: bool
    Description: Execute command in dry run mode
  --oci-version:
    Type: string
    Description: OCI image/artifact specification version (automatically determined by default)
  --resolve-image-digests:
    Type: bool
    Description: Pin image tags to digests
  --with-env:
    Type: bool
    Description: Include environment variables in the published OCI artifact
  -y, --yes:
    Type: bool
    Description: Assume "yes" as answer to all prompts
```

----------------------------------------

TITLE: Format Go Source Files with gofmt
DESCRIPTION: Ensures consistent code formatting for Go files using the `gofmt` tool. This command should be run on each changed file before committing to maintain a universal code style across the project.

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_1

LANGUAGE: Shell
CODE:
```
gofmt -s -w file.go
```

----------------------------------------

TITLE: docker compose config Command Options
DESCRIPTION: Lists all available command-line options for `docker compose config`, including their type, default value, and a brief description of their function.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_config.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose config
  Options:
    --dry-run (bool): Execute command in dry run mode
    --environment (bool): Print environment used for interpolation.
    --format (string): Format the output. Values: [yaml | json]
    --hash (string): Print the service config hash, one per line.
    --images (bool): Print the image names, one per line.
    --lock-image-digests (bool): Produces an override file with image digests
    --no-consistency (bool): Don't check model consistency - warning: may produce invalid Compose output
    --no-env-resolution (bool): Don't resolve service env files
    --no-interpolate (bool): Don't interpolate environment variables
    --no-normalize (bool): Don't normalize compose model
    --no-path-resolution (bool): Don't resolve file paths
    -o, --output (string): Save to file (default to stdout)
    --profiles (bool): Print the profile names, one per line.
    -q, --quiet (bool): Only validate the configuration, don't print anything
    --resolve-image-digests (bool): Pin image tags to digests
    --services (bool): Print the service names, one per line.
    --variables (bool): Print model variables and default values.
    --volumes (bool): Print the volume names, one per line.
```

----------------------------------------

TITLE: docker compose ls Command Options
DESCRIPTION: This section details the available command-line options for `docker compose ls`, including their types, default values, and descriptions. These options allow users to control the output and behavior of the command.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ls.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose ls

Description: Lists running Compose projects

Options:
  -a, --all (bool): Show all stopped Compose projects
  --dry-run (bool): Execute command in dry run mode
  --filter (filter): Filter output based on conditions provided
  --format (string, default: table): Format the output. Values: [table | json]
  -q, --quiet (bool): Only display project names
```

----------------------------------------

TITLE: Docker Compose General Command Syntax
DESCRIPTION: This snippet illustrates the general command-line syntax for invoking `docker compose`, including optional arguments, flags, and the structure for specifying subcommands and their respective arguments.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_0

LANGUAGE: text
CODE:
```
docker compose [-f <arg>...] [options] [COMMAND] [ARGS...]
```

----------------------------------------

TITLE: docker compose watch Command Options
DESCRIPTION: Details the available command-line options for the `docker compose watch` command, including their types, default values, and descriptions. These options allow users to control aspects like dry run execution, initial service startup, image pruning, and output verbosity.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_watch.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Options for 'docker compose watch':
- --dry-run:
    Type: bool
    Default: (none)
    Description: Execute command in dry run mode
- --no-up:
    Type: bool
    Default: (none)
    Description: Do not build & start services before watching
- --prune:
    Type: bool
    Default: true
    Description: Prune dangling images on rebuild
- --quiet:
    Type: bool
    Default: (none)
    Description: hide build output
```

----------------------------------------

TITLE: docker compose commit Command API Reference
DESCRIPTION: Detailed API documentation for the `docker compose commit` command, including its purpose and all available command-line options with their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_commit.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose commit
Description: Create a new image from a service container's changes

Options:
  -a, --author:
    Type: string
    Default: ""
    Description: Author (e.g., "John Hannibal Smith <hannibal@a-team.com>")
  -c, --change:
    Type: list
    Default: ""
    Description: Apply Dockerfile instruction to the created image
  --dry-run:
    Type: bool
    Default: ""
    Description: Execute command in dry run mode
  --index:
    Type: int
    Default: 0
    Description: index of the container if service has multiple replicas.
  -m, --message:
    Type: string
    Default: ""
    Description: Commit message
  -p, --pause:
    Type: bool
    Default: true
    Description: Pause container during commit
```

----------------------------------------

TITLE: Docker Compose Push Command-Line Options
DESCRIPTION: Reference for the available command-line options when using `docker compose push`, including their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_push.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Options:
  --dry-run:
    Type: bool
    Default: 
    Description: Execute command in dry run mode
  --ignore-push-failures:
    Type: bool
    Default: 
    Description: Push what it can and ignores images with push failures
  --include-deps:
    Type: bool
    Default: 
    Description: Also push images of services declared as dependencies
  -q, --quiet:
    Type: bool
    Default: 
    Description: Push without printing progress information
```

----------------------------------------

TITLE: Pretty-Print Docker Compose `ps` JSON Output with `jq`
DESCRIPTION: Illustrates how to use the `jq` utility to format and pretty-print the JSON output from `docker compose ps --format json`, enhancing readability for human inspection.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_7

LANGUAGE: console
CODE:
```
$ docker compose ps --format json | jq .
[
  {
    "ID": "1553b0236cf4d2715845f053a4ee97042c4f9a2ef655731ee34f1f7940eaa41a",
    "Name": "example-bar-1",
    "Command": "/docker-entrypoint.sh nginx -g 'daemon off;'",
    "Project": "example",
    "Service": "bar",
    "State": "exited",
    "Health": "",
    "ExitCode": 0,
    "Publishers": null
  },
  {
    "ID": "f02a4efaabb67416e1ff127d51c4b5578634a0ad5743bd65225ff7d1909a3fa0",
    "Name": "example-foo-1",
    "Command": "/docker-entrypoint.sh nginx -g 'daemon off;'",
    "Project": "example",
    "Service": "foo",
    "State": "running",
    "Health": "",
    "ExitCode": 0,
    "Publishers": [
      {
        "URL": "0.0.0.0",
        "TargetPort": 80,
        "PublishedPort": 8080,
        "Protocol": "tcp"
      }
    ]
  }
]
```

----------------------------------------

TITLE: docker compose attach Command Options
DESCRIPTION: Documents the command-line options available for the `docker compose attach` command, including their types, default values, and descriptions. This allows users to understand and configure the behavior of stream attachment.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_attach.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose attach:
  Description: Attach local standard input, output, and error streams to a service's running container
  Options:
    --detach-keys:
      Type: string
      Default: ""
      Description: Override the key sequence for detaching from a container.
    --dry-run:
      Type: bool
      Default: ""
      Description: Execute command in dry run mode
    --index:
      Type: int
      Default: "0"
      Description: index of the container if service has multiple replicas.
    --no-stdin:
      Type: bool
      Default: ""
      Description: Do not attach STDIN
    --sig-proxy:
      Type: bool
      Default: "true"
      Description: Proxy all received signals to the process
```

----------------------------------------

TITLE: Docker Compose CLI Options Reference
DESCRIPTION: This section provides a detailed reference for the command-line options used with Docker Compose. Each option includes its name, data type, default value (if applicable), and a comprehensive description of its purpose and behavior.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_2

LANGUAGE: APIDOC
CODE:
```
Option: -a, --all
  Type: bool
  Default: None
  Description: Show all stopped containers (including those created by the run command)

Option: --dry-run
  Type: bool
  Default: None
  Description: Execute command in dry run mode

Option: --filter
  Type: string
  Default: None
  Description: Filter services by a property (supported filters: status)

Option: --format
  Type: string
  Default: table
  Description: Format output using a custom template:
    'table': Print output in table format with column headers (default)
    'table TEMPLATE': Print output in table format using the given Go template
    'json': Print in JSON format
    'TEMPLATE': Print output using the given Go template.
    Refer to https://docs.docker.com/go/formatting/ for more information about formatting output with templates

Option: --no-trunc
  Type: bool
  Default: None
  Description: Don't truncate output

Option: --orphans
  Type: bool
  Default: true
  Description: Include orphaned services (not declared by project)

Option: -q, --quiet
  Type: bool
  Default: None
  Description: Only display IDs

Option: --services
  Type: bool
  Default: None
  Description: Display services
```

----------------------------------------

TITLE: Docker Compose Stats Command Options
DESCRIPTION: Documents the available command-line options for the `docker compose stats` command. Each option includes its name, type, and a detailed description of its functionality and impact on the command's output.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_stats.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose stats

Options:
  -a, --all (Type: bool, Default: N/A)
    Description: Show all containers (default shows just running)
  --dry-run (Type: bool, Default: N/A)
    Description: Execute command in dry run mode
  --format (Type: string, Default: N/A)
    Description: Format output using a custom template:
      'table': Print output in table format with column headers (default)
      'table TEMPLATE': Print output in table format using the given Go template
      'json': Print in JSON format
      'TEMPLATE': Print output using the given Go template.
      Refer to https://docs.docker.com/engine/cli/formatting/ for more information about formatting output with templates
  --no-stream (Type: bool, Default: N/A)
    Description: Disable streaming stats and only pull the first result
  --no-trunc (Type: bool, Default: N/A)
    Description: Do not truncate output
```

----------------------------------------

TITLE: docker compose cp Command Options
DESCRIPTION: Details the available command-line options for the `docker compose cp` command, including their types, default values, and descriptions. These options modify the behavior of the copy operation.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_cp.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose cp:
  Description: Copy files/folders between a service container and the local filesystem
  Options:
    --all:
      Type: bool
      Default: N/A
      Description: Include containers created by the run command
    -a, --archive:
      Type: bool
      Default: N/A
      Description: Archive mode (copy all uid/gid information)
    --dry-run:
      Type: bool
      Default: N/A
      Description: Execute command in dry run mode
    -L, --follow-link:
      Type: bool
      Default: N/A
      Description: Always follow symbol link in SRC_PATH
    --index:
      Type: int
      Default: 0
      Description: Index of the container if service has multiple replicas
```

----------------------------------------

TITLE: Docker Compose Command-Line Options Reference
DESCRIPTION: This section details the various command-line options that can be used with Docker Compose commands. Each option includes its name, data type, default value (if applicable), and a description of its purpose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
Docker Compose Command Options:
  --build (bool): Build image before starting container
  --cap-add (list): Add Linux capabilities
  --cap-drop (list): Drop Linux capabilities
  -d, --detach (bool): Run container in background and print container ID
  --dry-run (bool): Execute command in dry run mode
  --entrypoint (string): Override the entrypoint of the image
  -e, --env (stringArray): Set environment variables
  --env-from-file (stringArray): Set environment variables from file
  -i, --interactive (bool, default: true): Keep STDIN open even if not attached
  -l, --label (stringArray): Add or override a label
  --name (string): Assign a name to the container
  -T, --no-TTY (bool, default: true): Disable pseudo-TTY allocation (default: auto-detected)
  --no-deps (bool): Don't start linked services
  -p, --publish (stringArray): Publish a container's port(s) to the host
  --pull (string, default: policy): Pull image before running ("always"|"missing"|"never")
  -q, --quiet (bool): Don't print anything to STDOUT
  --quiet-build (bool): Suppress progress output from the build process
  --quiet-pull (bool): Pull without printing progress information
  --remove-orphans (bool): Remove containers for services not defined in the Compose file
  --rm (bool): Automatically remove the container when it exits
  -P, --service-ports (bool): Run command with all service's ports enabled and mapped to the host
  --use-aliases (bool): Use the service's network useAliases in the network(s) the container connects to
  -u, --user (string): Run as specified username or uid
  -v, --volume (stringArray): Bind mount a volume
  -w, --workdir (string): Working directory inside the container
```

----------------------------------------

TITLE: docker compose port Command API Reference
DESCRIPTION: Detailed API documentation for the `docker compose port` command, outlining its purpose and the various options available to control its behavior, including their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_port.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose port
Description: Prints the public port for a port binding

Options:
  --dry-run:
    Type: bool
    Default: N/A
    Description: Execute command in dry run mode
  --index:
    Type: int
    Default: 0
    Description: Index of the container if service has multiple replicas
  --protocol:
    Type: string
    Default: tcp
    Description: tcp or udp
```

----------------------------------------

TITLE: Docker Compose Restart Command API Reference
DESCRIPTION: Comprehensive API documentation for the `docker compose restart` command, including its primary function, implications for configuration changes, and a detailed list of all supported command-line options with their types and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_restart.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose restart
Description: Restarts all stopped and running services, or the specified services only.
  If you make changes to your `compose.yml` configuration, these changes are not reflected after running this command. For example, changes to environment variables (which are added after a container is built, but before the container's command is executed) are not updated after restarting.
  For configuring a service's restart policy, refer to the official documentation.

Options:
  --dry-run:
    Type: bool
    Description: Execute command in dry run mode
  --no-deps:
    Type: bool
    Description: Don't restart dependent services
  -t, --timeout:
    Type: int
    Default: 0
    Description: Specify a shutdown timeout in seconds
```

----------------------------------------

TITLE: docker compose pause Command Reference
DESCRIPTION: Detailed documentation for the `docker compose pause` command, including its purpose and available options.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_pause.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose pause
  Description: Pauses running containers of a service. They can be unpaused with `docker compose unpause`.
  Options:
    --dry-run:
      Type: bool
      Default:
      Description: Execute command in dry run mode
```

----------------------------------------

TITLE: Docker Compose Subcommands Reference
DESCRIPTION: This section provides a detailed reference of all available subcommands for `docker compose`. Each entry includes the subcommand's name and a brief description of its functionality, enabling users to understand the full range of operations supported by Docker Compose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Subcommands:
  attach: Attach local standard input, output, and error streams to a service's running container
  bridge: Convert compose files into another model
  build: Build or rebuild services
  commit: Create a new image from a service container's changes
  config: Parse, resolve and render compose file in canonical format
  cp: Copy files/folders between a service container and the local filesystem
  create: Creates containers for a service
  down: Stop and remove containers, networks
  events: Receive real time events from containers
  exec: Execute a command in a running container
  export: Export a service container's filesystem as a tar archive
  images: List images used by the created containers
  kill: Force stop service containers
  logs: View output from containers
  ls: List running compose projects
  pause: Pause services
  port: Print the public port for a port binding
  ps: List containers
  publish: Publish compose application
  pull: Pull service images
  push: Push service images
  restart: Restart service containers
  rm: Removes stopped service containers
  run: Run a one-off command on a service
  scale: Scale services
  start: Start services
  stats: Display a live stream of container(s) resource usage statistics
  stop: Stop services
  top: Display the running processes
  unpause: Unpause services
  up: Create and start containers
  version: Show the Docker Compose version information
  volumes: List volumes
  wait: Block until containers of all (or specified) services stop.
  watch: Watch build context for service and rebuild/refresh containers when files are updated
```

----------------------------------------

TITLE: Docker Compose Publish Command Options
DESCRIPTION: This section details the command-line options available for the `docker compose publish` command. It includes information on each option's name, data type, default value, and a brief description of its purpose and effect on the command's execution.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_publish.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose publish:
  Description: Publish compose application
  Options:
    --dry-run:
      Type: bool
      Default: ""
      Description: Execute command in dry run mode
    --oci-version:
      Type: string
      Default: ""
      Description: OCI image/artifact specification version (automatically determined by default)
    --resolve-image-digests:
      Type: bool
      Default: ""
      Description: Pin image tags to digests
    --with-env:
      Type: bool
      Default: ""
      Description: Include environment variables in the published OCI artifact
    -y, --yes:
      Type: bool
      Default: ""
      Description: Assume "yes" as answer to all prompts
```

----------------------------------------

TITLE: Executing Provider's Compose Up Command
DESCRIPTION: This console command illustrates how Compose invokes an external provider's `compose up` command. Compose passes the project name, service name, and translates `provider.options` into command-line flags (e.g., `--type`, `--size`). The provider is expected to use the project name to tag resources for later cleanup.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_1

LANGUAGE: console
CODE:
```
awesomecloud compose --project-name <NAME> up --type=mysql --size=256 "database"
```

----------------------------------------

TITLE: Execute Compose Metadata Subcommand
DESCRIPTION: Demonstrates how to invoke the `metadata` subcommand for a Compose extension to retrieve parameter information. This command takes no parameters and outputs a JSON structure.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_6

LANGUAGE: console
CODE:
```
awesomecloud compose metadata
```

----------------------------------------

TITLE: Docker Compose Kill Command Options Reference
DESCRIPTION: This section provides a detailed reference for all available options for the `docker compose kill` command, including their data types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_kill.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
docker compose kill options:
  --dry-run:
    type: bool
    description: Execute command in dry run mode
  --remove-orphans:
    type: bool
    description: Remove containers for services not defined in the Compose file
  -s, --signal:
    type: string
    default: SIGKILL
    description: SIGNAL to send to the container
```

----------------------------------------

TITLE: Docker Compose CLI Command-Line Options Reference
DESCRIPTION: This section details the available command-line options for `docker compose`, including their types, default values, and descriptions. These options control various aspects of Compose's behavior, such as resource inclusion, ANSI output, compatibility mode, and project configuration.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_2

LANGUAGE: APIDOC
CODE:
```
--all-resources:
  Type: bool
  Default: (empty)
  Description: Include all resources, even those not used by services

--ansi:
  Type: string
  Default: auto
  Description: Control when to print ANSI control characters ("never"|"always"|"auto")

--compatibility:
  Type: bool
  Default: (empty)
  Description: Run compose in backward compatibility mode

--dry-run:
  Type: bool
  Default: (empty)
  Description: Execute command in dry run mode

--env-file:
  Type: stringArray
  Default: (empty)
  Description: Specify an alternate environment file

-f, --file:
  Type: stringArray
  Default: (empty)
  Description: Compose configuration files

--parallel:
  Type: int
  Default: -1
  Description: Control max parallelism, -1 for unlimited

--profile:
  Type: stringArray
  Default: (empty)
  Description: Specify a profile to enable

--progress:
  Type: string
  Default: (empty)
  Description: Set type of progress output (auto, tty, plain, json, quiet)

--project-directory:
  Type: string
  Default: (default: the path of the, first specified, Compose file)
  Description: Specify an alternate working directory

-p, --project-name:
  Type: string
  Default: (empty)
  Description: Project name
```

----------------------------------------

TITLE: Docker Compose Export Command API Reference
DESCRIPTION: Detailed API documentation for the `docker compose export` command, including its purpose and available command-line options for controlling its behavior.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_export.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose export
Description: Export a service container's filesystem as a tar archive

Options:
  --dry-run (bool): Execute command in dry run mode
  --index (int, default: 0): index of the container if service has multiple replicas.
  -o, --output (string): Write to a file, instead of STDOUT
```

----------------------------------------

TITLE: docker compose unpause Command API Reference
DESCRIPTION: Detailed API documentation for the `docker compose unpause` command, including its functionality and available options.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_unpause.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose unpause
Description: Unpauses paused containers of a service
Options:
  --dry-run:
    Type: bool
    Default: 
    Description: Execute command in dry run mode
```

----------------------------------------

TITLE: docker compose alpha viz Command Options
DESCRIPTION: Details the available command-line options for the `docker compose alpha viz` command, including their types, default values, and descriptions. This command generates a Graphviz graph from a Compose file.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha_viz.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose alpha viz
Description: EXPERIMENTAL - Generate a graphviz graph from your compose file

Options:
  --dry-run:
    Type: bool
    Default: (none)
    Description: Execute command in dry run mode
  --image:
    Type: bool
    Default: (none)
    Description: Include service's image name in output graph
  --indentation-size:
    Type: int
    Default: 1
    Description: Number of tabs or spaces to use for indentation
  --networks:
    Type: bool
    Default: (none)
    Description: Include service's attached networks in output graph
  --ports:
    Type: bool
    Default: (none)
    Description: Include service's exposed ports in output graph
  --spaces:
    Type: bool
    Default: (none)
    Description: If given, space character ' ' will be used to indent,
otherwise tab character '\t' will be used
```

----------------------------------------

TITLE: docker compose pull Command Options Reference
DESCRIPTION: Comprehensive documentation of the command-line options available for `docker compose pull`, including their data types, default values, and a brief description of their functionality.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_pull.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose pull:
  Options:
    --dry-run:
      Type: bool
      Description: Execute command in dry run mode
    --ignore-buildable:
      Type: bool
      Description: Ignore images that can be built
    --ignore-pull-failures:
      Type: bool
      Description: Pull what it can and ignores images with pull failures
    --include-deps:
      Type: bool
      Description: Also pull services declared as dependencies
    --policy:
      Type: string
      Description: Apply pull policy ("missing"|"always")
    -q, --quiet:
      Type: bool
      Description: Pull without printing progress information
```

----------------------------------------

TITLE: docker compose top Command Options
DESCRIPTION: Available command-line options for the `docker compose top` command, detailing their type and purpose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_top.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose top
  Options:
    --dry-run:
      Type: bool
      Description: Execute command in dry run mode
```

----------------------------------------

TITLE: Docker Compose Alpha Scale Command Options
DESCRIPTION: This section documents the command-line options available for the 'docker compose alpha scale' command. It provides details on each option's name, type, default value, and a brief description of its function, enabling users to understand how to control the scaling behavior.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha_scale.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose alpha scale

Description: Scale services

Options:
  --dry-run:
    Type: (empty)
    Default: (empty)
    Description: Execute command in dry run mode
  --no-deps:
    Type: (empty)
    Default: (empty)
    Description: Don't start linked services
```

----------------------------------------

TITLE: Docker Compose Bridge Transformations List Command Reference
DESCRIPTION: Provides a comprehensive reference for the `docker compose bridge transformations list` command, including its aliases and all supported command-line options. This command is used to list available transformations.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_bridge_transformations_list.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose bridge transformations list

Purpose: List available transformations

Aliases:
  - docker compose bridge transformations list
  - docker compose bridge transformations ls

Options:
  --dry-run:
    Type: bool
    Default: N/A
    Description: Execute command in dry run mode
  --format:
    Type: string
    Default: table
    Description: Format the output. Values: [table | json]
  -q, --quiet:
    Type: bool
    Default: N/A
    Description: Only display transformer names
```

----------------------------------------

TITLE: Docker Compose RM Command Options
DESCRIPTION: Lists and describes the available command-line options for `docker compose rm`, including their types and effects.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_rm.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Options:
  --dry-run (bool): Execute command in dry run mode
  -f, --force (bool): Don't ask to confirm removal
  -s, --stop (bool): Stop the containers, if required, before removing
  -v, --volumes (bool): Remove any anonymous volumes attached to containers
```

----------------------------------------

TITLE: docker compose exec Command Options
DESCRIPTION: This section outlines the various command-line options available for `docker compose exec`, allowing users to customize how commands are executed within their Docker Compose services. Options include controlling detachment, environment variables, TTY allocation, and user context.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_exec.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose exec
Options:
  -d, --detach (Type: bool) [Default: ] - Detached mode: Run command in the background
  --dry-run (Type: bool) [Default: ] - Execute command in dry run mode
  -e, --env (Type: stringArray) [Default: ] - Set environment variables
  --index (Type: int) [Default: 0] - Index of the container if service has multiple replicas
  -T, --no-TTY (Type: bool) [Default: true] - Disable pseudo-TTY allocation. By default `docker compose exec` allocates a TTY.
  --privileged (Type: bool) [Default: ] - Give extended privileges to the process
  -u, --user (Type: string) [Default: ] - Run the command as this user
  -w, --workdir (Type: string) [Default: ] - Path to workdir directory for this command
```

----------------------------------------

TITLE: Docker Compose Volumes List Command Options
DESCRIPTION: Describes the command-line options available when listing Docker Compose volumes, including their types, default values, and detailed descriptions for controlling output and execution behavior.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_volumes.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose volumes list
Options:
  --dry-run:
    Type: bool
    Description: Execute command in dry run mode
  --format:
    Type: string
    Default: table
    Description: Format output using a custom template:
      'table': Print output in table format with column headers (default)
      'table TEMPLATE': Print output in table format using the given Go template
      'json': Print in JSON format
      'TEMPLATE': Print output using the given Go template.
      Refer to https://docs.docker.com/go/formatting/ for more information about formatting output with templates
  -q, --quiet:
    Type: bool
    Description: Only display volume names
```

----------------------------------------

TITLE: Docker Compose Watch Command Options
DESCRIPTION: Available command-line options for `docker compose alpha watch` to control its behavior, including dry run execution, skipping initial service startup, and suppressing build output.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_alpha_watch.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose alpha watch:
  Options:
    --dry-run:
      Description: Execute command in dry run mode
    --no-up:
      Description: Do not build & start services before watching
    --quiet:
      Description: hide build output
```

----------------------------------------

TITLE: Docker Compose Bridge CLI Reference
DESCRIPTION: Reference for the `docker compose bridge` command-line interface, including available subcommands and global options.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_bridge.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose bridge subcommands:
  - Name: convert
    Description: Convert compose files to Kubernetes manifests, Helm charts, or another model
  - Name: transformations
    Description: Manage transformation images
```

LANGUAGE: APIDOC
CODE:
```
docker compose bridge options:
  - Name: --dry-run
    Type: bool
    Default: (none)
    Description: Execute command in dry run mode
```

----------------------------------------

TITLE: Docker Compose Bridge Transformations CLI Reference
DESCRIPTION: Provides a reference for the 'docker compose bridge transformations' command, including its available subcommands ('create', 'list') and global options ('--dry-run'). This command set is used to manage transformation images within Docker Compose.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_bridge_transformations.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose bridge transformations:
  Description: Manage transformation images

  Subcommands:
    create:
      Description: Create a new transformation
    list:
      Description: List available transformations

  Options:
    --dry-run:
      Type: bool
      Default: null
      Description: Execute command in dry run mode
```

----------------------------------------

TITLE: List All Docker Compose Project Containers
DESCRIPTION: Shows all containers for a Docker Compose project, including both running and stopped ones. This provides a complete overview of all services defined in the project.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_5

LANGUAGE: console
CODE:
```
$ docker compose ps --all
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
example-bar-1   alpine    "/entrypoint.…"   bar        4 seconds ago   exited (0)
```

----------------------------------------

TITLE: Docker Compose Bridge Convert Command Options
DESCRIPTION: Details the `docker compose bridge convert` command, which converts Docker Compose files into various formats like Kubernetes manifests or Helm charts. This section outlines the available command-line options, their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_bridge_convert.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose bridge convert
Description: Convert compose files to Kubernetes manifests, Helm charts, or another model

Options:
  --dry-run (Type: bool, Default: N/A)
    Description: Execute command in dry run mode
  -o, --output (Type: string, Default: out)
    Description: The output directory for the Kubernetes resources
  --templates (Type: string, Default: N/A)
    Description: Directory containing transformation templates
  -t, --transformation (Type: stringArray, Default: docker/compose-bridge-kubernetes)
    Description: Transformation to apply to compose model
```

----------------------------------------

TITLE: docker compose stop Command Options
DESCRIPTION: Details the available command-line options for the `docker compose stop` command, including their types, default values, and descriptions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_stop.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose stop

Options:
  --dry-run:
    Type: bool
    Default: (none)
    Description: Execute command in dry run mode

  -t, --timeout:
    Type: int
    Default: 0
    Description: Specify a shutdown timeout in seconds
```

----------------------------------------

TITLE: Retrieving Docker System Information for Issue Reporting
DESCRIPTION: These commands are essential for gathering detailed information about your Docker environment, including version, context, and system-wide details. This data is crucial for debugging and accurately reporting issues to the Docker maintainers.

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_0

LANGUAGE: Shell
CODE:
```
docker version
```

LANGUAGE: Shell
CODE:
```
docker context show
```

LANGUAGE: Shell
CODE:
```
docker info
```

----------------------------------------

TITLE: Define Multi-Container Application Services with Docker Compose YAML
DESCRIPTION: This YAML configuration defines two services, 'web' and 'redis', for a multi-container application. The 'web' service builds from the current directory, maps port 5000, and mounts the current directory as a volume. The 'redis' service uses the official Redis image, demonstrating how to define interconnected services in a Compose file.

SOURCE: https://github.com/docker/compose/blob/main/README.md#_snippet_0

LANGUAGE: yaml
CODE:
```
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - .:/code
  redis:
    image: redis
```

----------------------------------------

TITLE: Compose Extension Metadata JSON Output Schema
DESCRIPTION: Defines the expected JSON structure returned by the `metadata` subcommand. The top-level elements include a `description` of the provider and objects for `up` and `down` commands. Each command object contains a `parameters` array, where each parameter specifies its `name`, `description`, `required` status (boolean), `type` (e.g., 'string', 'integer', 'boolean'), and optional `default` or `enum` values.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_7

LANGUAGE: json
CODE:
```
{
  "description": "Manage services on AwesomeCloud",
  "up": {
    "parameters": [
      {
        "name": "type",
        "description": "Database type (mysql, postgres, etc.)",
        "required": true,
        "type": "string"
      },
      {
        "name": "size",
        "description": "Database size in GB",
        "required": false,
        "type": "integer",
        "default": "10"
      },
      {
        "name": "name",
        "description": "Name of the database to be created",
        "required": true,
        "type": "string"
      }
    ]
  },
  "down": {
    "parameters": [
      {
        "name": "name",
        "description": "Name of the database to be removed",
        "required": true,
        "type": "string"
      }
    ]
  }
}
```

----------------------------------------

TITLE: Override and Extend Services with `compose.admin.yaml`
DESCRIPTION: This YAML snippet shows how `compose.admin.yaml` can override and extend a service defined in a preceding Compose file. It demonstrates adding a build context and an environment variable to the `webapp` service, showcasing how layered configurations modify existing definitions.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_5

LANGUAGE: yaml
CODE:
```
services:
  webapp:
    build: .
    environment:
      - DEBUG=1
```

----------------------------------------

TITLE: Docker Compose Events Command Options
DESCRIPTION: This section documents the command-line options available for the `docker compose events` command. It specifies each option's name, data type, and a brief description of its functionality.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_events.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
docker compose events options:
  --dry-run:
    type: bool
    description: Execute command in dry run mode
  --json:
    type: bool
    description: Output events as a stream of json objects
```

----------------------------------------

TITLE: List Running Docker Compose Containers
DESCRIPTION: Shows currently running containers for a Docker Compose project. This command provides an overview of active services, including their name, image, command, service name, creation time, current status, and exposed network ports.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_0

LANGUAGE: console
CODE:
```
$ docker compose ps
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
```

----------------------------------------

TITLE: Run Docker Compose Service with Manual Port Mapping
DESCRIPTION: This command shows how to manually map specific host ports to container ports using `--publish` or `-p` options, similar to `docker run`. It allows precise control over port exposure for the executed command. This is ideal for testing specific port configurations or when the service's default port mappings are insufficient.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_9

LANGUAGE: console
CODE:
```
docker compose run --publish 8080:80 -p 2022:22 -p 127.0.0.1:2021:21 web python manage.py shell
```

----------------------------------------

TITLE: Docker Compose Bridge Transformation Creation Options
DESCRIPTION: Defines the available command-line options for creating a new transformation in docker compose bridge, including dry-run and source transformation specification.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_bridge_transformations_create.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Command: docker compose bridge transformations create
Description: Create a new transformation

Options:
  --dry-run (Type: bool): Execute command in dry run mode
  -f, --from (Type: string, Default: docker/compose-bridge-kubernetes): Existing transformation to copy
```

----------------------------------------

TITLE: List All Docker Compose Containers (Running and Stopped)
DESCRIPTION: Displays all containers, including those that have exited, for a Docker Compose project using the `--all` flag. This provides a comprehensive view of all services defined in the project, regardless of their current operational status.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_1

LANGUAGE: console
CODE:
```
$ docker compose ps --all
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
example-bar-1   alpine    "/entrypoint.…"   bar        4 seconds ago   exited (0)
```

----------------------------------------

TITLE: Executing docker compose pull for a Specific Service
DESCRIPTION: Illustrates the execution of the `docker compose pull db` command to pull the image for the 'db' service as defined in a `compose.yaml` file, along with the typical console output showing the image pulling progress.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_pull.md#_snippet_2

LANGUAGE: console
CODE:
```
$ docker compose pull db
[+] Running 1/15
 ⠸ db Pulling                                                             12.4s
   ⠿ 45b42c59be33 Already exists                                           0.0s
   ⠹ 40adec129f1a Downloading  3.374MB/4.178MB                             9.3s
   ⠹ b4c431d00c78 Download complete                                        9.3s
   ⠹ 2696974e2815 Download complete                                        9.3s
   ⠹ 564b77596399 Downloading  5.622MB/7.965MB                             9.3s
   ⠹ 5044045cf6f2 Downloading  216.7kB/391.1kB                             9.3s
   ⠹ d736e67e6ac3 Waiting                                                  9.3s
   ⠹ 390c1c9a5ae4 Waiting                                                  9.3s
   ⠹ c0e62f172284 Waiting                                                  9.3s
   ⠹ ebcdc659c5bf Waiting                                                  9.3s
   ⠹ 29be22cb3acc Waiting                                                  9.3s
   ⠹ f63c47038e66 Waiting                                                  9.3s
   ⠹ 77a0c198cde5 Waiting                                                  9.3s
   ⠹ c8752d5b785c Waiting                                                  9.3s
```

----------------------------------------

TITLE: Provider-Compose Communication Sequence Diagram
DESCRIPTION: This Mermaid sequence diagram visualizes the interaction flow between the Shell, Compose, and an external Provider during the `docker compose up` lifecycle. It shows Compose invoking the provider, the provider sending JSON messages (info, setenv) back to Compose, and Compose relaying information to the shell or setting environment variables for dependent services.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_3

LANGUAGE: mermaid
CODE:
```
sequenceDiagram
    Shell->>Compose: docker compose up
    Compose->>Provider: compose up --project-name=xx --foo=bar "database"
    Provider--)Compose: json { "info": "pulling 25%" }
    Compose-)Shell: pulling 25%
    Provider--)Compose: json { "info": "pulling 50%" }
    Compose-)Shell: pulling 50%
    Provider--)Compose: json { "info": "pulling 75%" }
    Compose-)Shell: pulling 75%
    Provider--)Compose: json { "setenv": "URL=http://cloud.com/abcd:1234" }
    Compose-)Compose: set DATABASE_URL
    Provider-)Compose: EOF (command complete) exit 0
    Compose-)Shell: service started
```

----------------------------------------

TITLE: docker compose logs Command Options Reference
DESCRIPTION: This section details the various command-line options available for the 'docker compose logs' command, including their data types, default values, and a brief description of their functionality. These options allow users to control how logs are displayed, filtered by time, or limited by the number of lines.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_logs.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose logs:
  Options:
    --dry-run:
      Type: bool
      Default: 
      Description: Execute command in dry run mode
    -f, --follow:
      Type: bool
      Default: 
      Description: Follow log output
    --index:
      Type: int
      Default: 0
      Description: index of the container if service has multiple replicas
    --no-color:
      Type: bool
      Default: 
      Description: Produce monochrome output
    --no-log-prefix:
      Type: bool
      Default: 
      Description: Don't print prefix in logs
    --since:
      Type: string
      Default: 
      Description: Show logs since timestamp (e.g. 2013-01-02T13:23:37Z) or relative (e.g. 42m for 42 minutes)
    -n, --tail:
      Type: string
      Default: all
      Description: Number of lines to show from the end of the logs for each container
    -t, --timestamps:
      Type: bool
      Default: 
      Description: Show timestamps
    --until:
      Type: string
      Default: 
      Description: Show logs before a timestamp (e.g. 2013-01-02T13:23:37Z) or relative (e.g. 42m for 42 minutes)
```

----------------------------------------

TITLE: Defining a Service with an External Provider in Compose YAML
DESCRIPTION: This YAML snippet demonstrates how to define a service in a `compose.yaml` file that is managed by an external provider. The `provider` attribute specifies the `type` of the provider (e.g., `awesomecloud`) and `options` specific to that provider, such as the database type, size, and name. This allows Compose to delegate the management of this service to a third-party tool.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_0

LANGUAGE: yaml
CODE:
```
  database:
    provider:
      type: awesomecloud
      options:
        type: mysql
        size: 256
        name: myAwesomeCloudDB
```

----------------------------------------

TITLE: docker compose down Command Options
DESCRIPTION: Documents the available command-line options for the `docker compose down` command, including their type, default values, and a brief description of their functionality. These options allow for fine-grained control over the removal process.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_down.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
docker compose down Options:
  --dry-run (Type: bool)
    Description: Execute command in dry run mode
  --remove-orphans (Type: bool)
    Description: Remove containers for services not defined in the Compose file
  --rmi (Type: string)
    Description: Remove images used by services. "local" remove only images that don't have a custom tag ("local"\|"all")
  -t, --timeout (Type: int, Default: 0)
    Description: Specify a shutdown timeout in seconds
  -v, --volumes (Type: bool)
    Description: Remove named volumes declared in the "volumes" section of the Compose file and anonymous volumes attached to containers
```

----------------------------------------

TITLE: Git Branch and Commit Management for Contributions
DESCRIPTION: Provides essential Git commands for managing branches and commits during the contribution process. This includes updating pull requests cleanly with `rebase` and squashing commits into logical units of work for a clean and concise history.

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_2

LANGUAGE: Shell
CODE:
```
rebase master
```

LANGUAGE: Shell
CODE:
```
merge master
```

LANGUAGE: Shell
CODE:
```
git rebase -i
```

LANGUAGE: Shell
CODE:
```
git push -f
```

----------------------------------------

TITLE: Filter Docker Compose Containers by Status using `--filter`
DESCRIPTION: Demonstrates using the more general `--filter` flag to achieve the same result as `--status=running`, filtering containers to show only those in a 'running' state.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_10

LANGUAGE: console
CODE:
```
$ docker compose ps --filter status=running
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
```

----------------------------------------

TITLE: List Running Docker Compose Project Containers
DESCRIPTION: Displays only the currently running containers associated with a Docker Compose project, along with their image, command, service name, creation time, status, and exposed ports.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_4

LANGUAGE: console
CODE:
```
$ docker compose ps
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
```

----------------------------------------

TITLE: Run Command with Manual Port Mapping in Docker Compose
DESCRIPTION: This command runs `python manage.py shell` in the `web` service container, explicitly mapping specific ports from the container to the host. It uses the `--publish` or `-p` options, similar to `docker run`, to define custom port forwarding rules for the one-time command.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_2

LANGUAGE: console
CODE:
```
$ docker compose run --publish 8080:80 -p 2022:22 -p 127.0.0.1:2021:21 web python manage.py shell
```

----------------------------------------

TITLE: Run Bash in Docker Compose Web Service
DESCRIPTION: This snippet demonstrates how to execute a one-time `bash` command within the `web` service container using `docker compose run`. The command provided (`bash`) overrides any default command specified in the service's configuration. This is useful for interactive debugging or quick administrative tasks.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_7

LANGUAGE: console
CODE:
```
docker compose run web bash
```

----------------------------------------

TITLE: Automating Git Commit Sign-off with -s Flag
DESCRIPTION: A Git command to automatically add the 'Signed-off-by' line to a commit message. This feature relies on the user's `user.name` and `user.email` Git configurations to populate the sign-off information.

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_5

LANGUAGE: bash
CODE:
```
git commit -s
```

----------------------------------------

TITLE: Run Command with Service Ports in Docker Compose
DESCRIPTION: This command executes `python manage.py shell` within the `web` service container, ensuring that all ports defined in the service configuration are created and mapped to the host. This is useful when the command requires network access via the service's defined ports, unlike the default behavior of `docker compose run`.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_1

LANGUAGE: console
CODE:
```
$ docker compose run --service-ports web python manage.py shell
```

----------------------------------------

TITLE: Provider Communication: Info Message Format
DESCRIPTION: This JSON snippet shows the required format for providers to communicate status updates to Compose via `stdout`. Messages must include a `type` (e.g., `info`, `error`, `setenv`, `debug`) and a `message` attribute. `info` messages are rendered by Compose as service state updates in the progress UI.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_2

LANGUAGE: json
CODE:
```
{ "type": "info", "message": "preparing mysql ..." }
```

----------------------------------------

TITLE: Disable Docker Compose Helper Menu
DESCRIPTION: Shows how to disable the interactive helper menu that appears when running `docker compose up` in attached mode, using the `--menu=false` flag.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose.md#_snippet_10

LANGUAGE: console
CODE:
```
docker compose up --menu=false
```

----------------------------------------

TITLE: Run Command and Remove Container After Execution in Docker Compose
DESCRIPTION: This command runs a database upgrade script (`python manage.py db upgrade`) within the `web` service container. The `--rm` flag ensures that the container is automatically removed once the command finishes, overriding any defined restart policy for cleanup.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_5

LANGUAGE: console
CODE:
```
$ docker compose run --rm web python manage.py db upgrade
```

----------------------------------------

TITLE: Remove Stopped Docker Compose Service Container
DESCRIPTION: Demonstrates the interactive removal of a stopped service container using `docker compose rm`, including the confirmation prompt.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_rm.md#_snippet_0

LANGUAGE: console
CODE:
```
$ docker compose rm
Going to remove djangoquickstart_web_run_1
Are you sure? [yN] y
Removing djangoquickstart_web_run_1 ... done
```

----------------------------------------

TITLE: Provider Communication: Set Environment Variable Message
DESCRIPTION: This JSON snippet shows how a provider can instruct Compose to set an environment variable for dependent services. When a `setenv` message is received, Compose automatically prefixes the variable name with the dependent service's name (e.g., `DATABASE_URL` for a service named `database` that depends on it). This mechanism allows providers to expose connection details or other necessary configurations.

SOURCE: https://github.com/docker/compose/blob/main/docs/extension.md#_snippet_5

LANGUAGE: json
CODE:
```
{"type": "setenv", "message": "URL=https://awesomecloud.com/db:1234"}
```

----------------------------------------

TITLE: Developer Certificate of Origin (DCO) Specification
DESCRIPTION: The Developer Certificate of Origin (DCO) version 1.1, a legal document that contributors must agree to when submitting patches to open-source projects. It certifies the origin and licensing of contributions, ensuring compliance with project licensing terms.

SOURCE: https://github.com/docker/compose/blob/main/CONTRIBUTING.md#_snippet_3

LANGUAGE: text
CODE:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

----------------------------------------

TITLE: Output Docker Compose `ps` Data in JSON Format
DESCRIPTION: Changes the output format of the `docker compose ps` command to JSON, making it suitable for parsing by other tools or scripts. This snippet shows the raw JSON array output.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_6

LANGUAGE: console
CODE:
```
$ docker compose ps --format json
[{"ID":"1553b0236cf4d2715845f053a4ee97042c4f9a2ef655731ee34f1f7940eaa41a","Name":"example-bar-1","Command":"/docker-entrypoint.sh nginx -g 'daemon off;'","Project":"example","Service":"bar","State":"exited","Health":"","ExitCode":0,"Publishers":null},{"ID":"f02a4efaabb67416e1ff127d51c4b5578634a0ad5743bd65225ff7d1909a3fa0","Name":"example-foo-1","Command":"/docker-entrypoint.sh nginx -g 'daemon off;'","Project":"example","Service":"foo","State":"running","Health":"","ExitCode":0,"Publishers":[{"URL":"0.0.0.0","TargetPort":80,"PublishedPort":8080,"Protocol":"tcp"}]}]
```

----------------------------------------

TITLE: Run Docker Compose Command and Remove Container
DESCRIPTION: This command utilizes the `--rm` flag to automatically remove the container after the specified command (`python manage.py db upgrade`) finishes execution. This flag also overrides any restart policy defined for the service. It's particularly useful for one-off tasks like database migrations or build processes where the container is no longer needed afterward.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_run.md#_snippet_12

LANGUAGE: console
CODE:
```
docker compose run --rm web python manage.py db upgrade
```

----------------------------------------

TITLE: Filter Docker Compose Containers by Running Status
DESCRIPTION: Filters the displayed containers to include only those that are currently in a 'running' state, providing a focused view of active services.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_8

LANGUAGE: console
CODE:
```
$ docker compose ps --status=running
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-foo-1   alpine    "/entrypoint.…"   foo        4 seconds ago   Up 2 seconds    0.0.0.0:8080->80/tcp
```

----------------------------------------

TITLE: Filter Docker Compose Containers by Exited Status
DESCRIPTION: Filters the displayed containers to include only those that have 'exited', useful for inspecting services that have completed their execution or failed.

SOURCE: https://github.com/docker/compose/blob/main/docs/reference/compose_ps.md#_snippet_9

LANGUAGE: console
CODE:
```
$ docker compose ps --status=exited
NAME            IMAGE     COMMAND           SERVICE    CREATED         STATUS          PORTS
example-bar-1   alpine    "/entrypoint.…"   bar        4 seconds ago   exited (0)
```