import argparse
import os
import random
import string
import pykrylov
from pykrylov.util.consts import EXP_ID
import subprocess

root_dir = os.path.dirname(os.path.abspath(__file__))
MASTER_PORT = 2020


def init_krylov_context():
    if "KRYLOV_WS_NAME" not in os.environ:
        # Get params
        context = pykrylov.util.get_task_context()
        if "experiment_id" in context:
            experiment_id = context["experiment_id"]
            # These 2 lines make task logs viewable via experiment view on aihub
            pykrylov.util.set_global_context({EXP_ID: experiment_id})
            pykrylov.ems.experiment.update_experiment(
                experiment_id,
                runtime={"workflow": {"runId": os.environ["KRYLOV_WF_RUN_ID"]}},
            )
        os.environ["TRANSFORMERS_CACHE"] = (
            f"/data/{os.environ['KRYLOV_NAMESPACE']}/data/{os.environ['KRYLOV_PRINCIPAL']}/.llm_cache/hf_cache"
        )
        os.environ["TORCH_EXTENSIONS_DIR"] = (
            f"/data/{os.environ['KRYLOV_NAMESPACE']}/data/{os.environ['KRYLOV_PRINCIPAL']}/.llm_cache/torch_cache"
        )
        os.environ["HF_MODULES_CACHE"] = (
            f"/data/{os.environ['KRYLOV_NAMESPACE']}/data/{os.environ['KRYLOV_PRINCIPAL']}/.llm_cache/hf_module_cache"
        )
        return {
            "gpu_per_node": int(context["gpu_per_node"]),
            "num_nodes": int(context["num_nodes"]),
            "script": context["script"],
        }


def get_distributed_info():
    context = pykrylov.util.get_task_context()
    master_name = context["master_name"]
    master_service_name = context["master_service_name"]
    # Get master IP address from krylov environment variables
    print(f"Getting a list of ip addresses for of service {master_name}")
    ip_list = pykrylov.distributed.get_ip_list(master_name, master_service_name)
    print(f"Got a list of ips: {' '.join(ip_list)}")
    master_addr = ip_list[0]  # master rank always 0
    # Get rank from krylov task index environment variable
    print(f"Getting rank of this task")
    rank = pykrylov.distributed.get_task_index()
    print(f"Rank of this task is {rank}")
    return {
        "ip_list": ip_list,
        "master_addr": master_addr,
        "rank": rank,
    }


def run_fn():
    context = init_krylov_context()
    if context["num_nodes"] > 1:
        dist_info = get_distributed_info()
        os.environ["MASTER_ADDR"] = dist_info["master_addr"]
        os.environ["NODE_RANK"] = str(dist_info["rank"])

    script_path = os.path.join(root_dir, context["script"])
    os.chmod(script_path, 755)
    output = subprocess.run(script_path, check=True)
    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")


def submit_task():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Which script to run")
    parser.add_argument(
        "--ems_project", default="mnist-baliao", help="EMS project name"
    )
    parser.add_argument(
        "--experiment_name", default="llama3-cpt", help="Experiment name"
    )
    parser.add_argument("--cluster", default="tess137", help="Krylov cluster")
    parser.add_argument("-n", "--namespace", default="chatgpt", type=str)
    parser.add_argument(
        "-i",
        "--image",
        default="ecr.vip.ebayc3.com/baliao/apiq:omniquant_autogptq_bitsandbytes_peft0.11.2dev0_other",
        help="Docker image",
    )
    parser.add_argument("--cpu", default=16, help="Use cpus")
    parser.add_argument("--memory", default=512, help="Use memory")
    parser.add_argument(
        "--gpu_per_node", default=1, type=int, help="How many GPUs per node"
    )
    parser.add_argument("--num_nodes", default=4, type=int, help="How many nodes")
    parser.add_argument(
        "--rack_name", default=None, type=str, help="Specify which rack to use"
    )
    parser.add_argument("--pvc", action="store_true", help="Mount pvc")
    # parser.add_argument('--model', default='phase1_v0', help='Which model to run')
    args = parser.parse_args()

    if args.rack_name is not None:
        assert args.rack_name in ["slc_slc03_01-0200_11_20", "slc_slc03_01-0200_12_20"]

    # pykrylov.util.switch_krylov(args.cluster)

    if not args.cluster == "tess137":
        session = pykrylov.Session(namespace=args.namespace)

    experiment_id = None
    experiment_name = args.experiment_name
    if args.ems_project:
        if experiment_name is None:
            experiment_name = "llm"
        if not args.cluster == "tess137":
            experiment_id = pykrylov.ems.experiment.create_experiment(
                args.ems_project, experiment_name
            )

    master_name = "llm_" + "".join(random.choices(string.ascii_letters, k=8))
    master_service_name = master_name + "_svc"

    if args.num_nodes > 1:
        task = pykrylov.distributed.DistributedTask(
            run_fn,
            args=[],
            docker_image=args.image,
            parallelism=args.num_nodes,
            name=master_name,
            service_name=master_service_name,
            service_port=MASTER_PORT,
        )
    else:
        task = pykrylov.Task(
            run_fn,
            args=[],
            docker_image=args.image,
        )

    task.add_task_parameters(
        {
            "ems_project": args.ems_project,
            "experiment_name": experiment_name,
            "gpu_per_node": args.gpu_per_node,
            "num_nodes": args.num_nodes,
            "master_name": master_name,
            "master_service_name": master_service_name,
            "master_port": MASTER_PORT,
            "experiment_id": experiment_id,
            "script": os.path.basename(args.script),
        }
    )

    task.add_cpu(args.cpu)
    task.add_memory(args.memory)
    if args.gpu_per_node > 0:
        if args.cluster == "tess137":
            task.add_execution_parameter(
                "accelerator", {"type": "gpu", "quantity": str(args.gpu_per_node)}
            )
            # task.run_on_gpu(args.gpu_per_node, model='a100')
        else:
            task.run_on_gpu(args.gpu_per_node)

    if args.cluster == "tess40":
        task.mount_nfs("mtrepo", "10.5.1.56", "/krylov_shared_volume/krylov_shared")
    if args.cluster == "tess137":
        task.mount_pvc("mtrepo", "nlp-ebert-01", args.cluster)
        # task.mount_pvc('nushare', 'krylov-user-pvc-nlp-137', args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
    if args.cluster == "tess45":
        task.mount_pvc("nushare", "krylov-user-pvc-nlp-45", args.cluster)
        # task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)

    # task.add_directory(os.path.join(root_dir, "playground"))
    print("root_dir", root_dir)
    print("args.script", args.script)
    # task.add_directory(os.path.join(root_dir, "scripts"))
    task.add_directory(root_dir)
    # task.add_directory(os.path.join(root_dir, "sam2"))
    task.add_file(args.script)

    if args.cluster == "tess38":
        print("here")
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
        task.mount_pvc("nushare", "krylov-user-pvc-nlp-38", args.cluster)
        task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)
        if args.gpu_per_node > 0:
            task.add_execution_parameter(
                "accelerator",
                {
                    "type": "gpu",
                    "quantity": str(args.gpu_per_node),
                    "labels": {"family": "tesla", "product": "nvidia", "model": "a100"},
                },
            )
        task.add_execution_parameter("enableChooseCluster", "true")

    if args.cluster == "tess137":

        # 2023 09 19
        task.add_execution_parameter("requireSameRack", "true")
        from collections import OrderedDict

        workflow = pykrylov.Flow(task)
        workflow.execution_parameters.add_execution_parameter(
            "enableChooseCluster", "true"
        )
        # task.add_execution_parameter('enableChooseCluster', 'true')

    # specify rack
    if args.rack_name is not None:
        task.add_execution_parameter(
            "nodeSelector", {"failure-domain.tess.io/rack": args.rack_name}
        )

    if args.cluster == "tess137":
        # TODO does it really return run_id or experiment_id?
        run_id = pykrylov.Session(namespace=args.namespace).submit_experiment(
            workflow, args.ems_project, args.experiment_name
        )
    else:
        run_id = session.submit(task)

        if experiment_id:
            message = pykrylov.ems.attributes.add_runtime(
                experiment_id,
                {
                    "workflow": {
                        "runId": run_id,
                    }
                },
            )
            print(message)

    print(f"Submitted a task with run_id {run_id}")
    if experiment_id:
        link = f"https://{args.cluster[-2:]}.aihub.krylov.vip.ebay.com/projects/{args.ems_project}/experiments/{experiment_id}/info"
        print(
            f"You can monitor progress and download translation result by visiting {link}"
        )


if __name__ == "__main__":
    submit_task()

# python submit.py mono_ft.sh --gpu_per_node 8  --num_nodes 1 --cpu 32 --memory 512 --image ecr.vip.ebayc3.com/baliao/apiq:omniquant_autogptq_bitsandbytes_peft0.11.2dev0_other --experiment_name llama3-cpt --pvc
