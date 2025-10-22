# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entrypoint script for cloud ml base docker image.
"""

import argparse
import datetime
import glob
import json
import logging
import multiprocessing
import os
import pipes
import re
import resource
import signal
import sys
import threading
import time
import traceback

from builtins import str  # pylint: disable=redefined-builtin
import httplib2
from oauth2client.client import GoogleCredentials
from pkg_resources import parse_version
from six import iteritems
from six import reraise

# pylint: disable=g-import-not-at-top
try:
  import subprocess32 as subprocess
except ImportError:
  import subprocess


if sys.version_info.major <= 2:
  python_version = 'python'
  pip_version = 'pip'
else:
  python_version = 'python3'
  pip_version = 'pip3'


class CredentialsRefreshError(Exception):
  """Wrapper of errors thrown when refreshes credentials."""
  pass


class TpuIpResolveError(Exception):
  """Wrapper of errors thrown when resolving TPU node IP fails."""
  pass


class CudaInitError(Exception):
  """Wrapper of errors thrown when CUDA was not Initialized correctly."""
  pass


# pylint: disable=line-too-long
# Format of a TensorFlow C++ log.
# See: https://github.com/tensorflow/tensorflow/blob/c8b59c046895fa5b6d79f73e0b5817330fcfbfc1/tensorflow/core/platform/default/logging.cc#L79
TENSORFLOW_CPP_LOG_FORMAT = re.compile(
    r'(?:\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{6}:\s)*'
    r'([DIWEF])\s+([^: \]]+):(\d+)] (.*)')
# Format of a grpc log.
# See: https://github.com/grpc/grpc/blob/d0fbba52d6e379b76a69016bc264b96a2318315f/src/core/lib/support/log_posix.c#L96
GRPC_LOG_FORMAT = re.compile(
    r'([DIWEF])(\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{9})\s+(\d+)\s+'
    r'([^: \]]+):(\d+)] (.*)')
# Default termination log file path, uses this path if
# env variable TERMINATION_LOG is not set.
DEFAULT_TERMINATION_LOG_PATH = '/dev/termination-log'
# Default value of env variable LOG_FILE_TO_WRITE, which is the file where
# sitecustomize writes its logs, puts ERROR_SENTINEL under
# the same dir as this file.
# Note that this may be overridden (currently with the same value) by
# //cloud/ml/gke/tensorflow_runtime_manager.cc
DEFAULT_LOG_FILE_TO_WRITE = '/var/log-storage/output.log'
# Default error sentinenl filename. uses this value if env variables
# ERROR_SENTINEL is not set.
DEFAULT_ERROR_SENTINEL_FILENAME = 'failed.sentinel'

# Assumes all valid traceback should start with this
DEFAULT_TRACEBACK_STARTING_LINE = 'Traceback (most recent call last):'

# The file path where sitecustomize.py writes the last error stacktrace.
TRACEBACK_FILE_PATH = '/tmp/last_traceback.log'

# The overall size of the termination log can't be larger than 4K, so
# limiting the traceback part to 2K.
MAX_TRACEBACK_SIZE_BYTES = 2 * 1024

# When something goes wrong with traceback processing, this message will
# replace the actual traceback.
NO_TRACEBACK_MESSAGE = '(no traceback available)'

# A placeholder to put in place of deleted lines when traceback gets truncated.
TRACEBACK_TRUNCATION_SIGN = '  [...]'

GPU_USAGE_POLL_INTERVAL_IN_SECONDS = 300

TPU_JOB_RETRY_COUNT = 240

# pylint: enable=line-too-long


class _FailureHandler(object):
  """Handles failures in training.

  If core files are available, invoke gdb to get print stack
  trace. Then copy core files to GCS.
  """

  def __init__(self,
               command_working_directory,
               project_id,
               ignore_core_files_older_than=None):
    """Configures _FailureHandler.

    Args:
      command_working_directory: The directory in which command was run.
        This is used to find the core dump files.
      project_id: Cloud project ID.
      ignore_core_files_older_than: Optional variable, if specified,
        is a datetime.datetime object specifying timestamp of oldest
        core file to look for. This parameter is useful for ignoring
        core files that existed prior to the running of the command.
    """
    self._directory = command_working_directory or os.getcwd()
    self._project_id = project_id
    self._ignore_core_files_older_than = ignore_core_files_older_than

  def on_failure(self, core_dump_glob_patterns):
    core_file = self._find_coredump(core_dump_glob_patterns)
    if core_file:
      self._backtrace(core_file)

  def _bucket_name(self):
    """Generate a usable bucket name from the project id.

    Enforces rules from: https://cloud.google.com/storage/docs/naming

    Returns:
      String bucket name without backslashes at either end.
    """

    if not self._project_id:

      # TODO(jlewi): send logs to shadow project instead of user. b/30262794.
      logging.warning('Unable to get project id.  Will not make bucket.')
      return None

    bucket = ('%s-coredumps' % self._project_id).lower()
    bucket = re.sub(r'[^a-z0-9-]', '-', bucket)
    bucket = re.sub(r'^[^a-z]', '', bucket)
    bucket = re.sub(r'goog', '', bucket)

    return bucket

  def _make_bucket(self, bucket):
    """Create a GCS bucket for coredump storage if it doesn't already exist.

    Args:
      bucket: Name of bucket to be used.  Assumed to be cleaned.
    Returns:
      String bucket name without backslashes at either end or None if failure.
    """

    command = ['gsutil', '-q', 'mb', '-p', self._project_id, 'gs://' + bucket]

    # Ignore the returncode for bucket creation, because it 'fails' if the
    # bucket already exists.
    _ = self._call(command)

    return bucket

  # TODO(joshgc) use VM name from METADATA.
  @staticmethod
  def _docker_id():
    """Retrieve a Docker VM instance id, if it exists."""

    val = 'DockerIDNotFound'
    try:
      with open('/proc/self/cgroup') as ff:
        for line in ff:
          pieces = line.split(':')
          if pieces[1].lower() == 'cpu':
            val = pieces[2]
            break
    except IOError:
      pass

    val = re.sub(r'[^a-zA-Z0-9_]', '', val)
    return val

  @staticmethod
  def _call(command):
    """Create a subprocess.  Do not throw exception on failure."""

    proc = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()

    command_str = ' '.join(pipes.quote(s) for s in command)
    prefix = 'Called "%s".' % command_str
    if proc.returncode != 0:
      logging.warning('%s Stdout was <%s>', prefix, out)
      logging.warning('%s Stderr was <%s>', prefix, err)

    return proc.returncode

  def _find_coredump(self, patterns=('core', 'core.*')):
    """Look in current working directory for core or core.* files."""

    paths = []
    for pattern in patterns:
      for path in glob.glob(os.path.join(self._directory, pattern)):
        if (not self._ignore_core_files_older_than or
            self._ignore_core_files_older_than <=
            datetime.datetime.fromtimestamp(os.path.getmtime(path))):
          paths.append(path)
    if paths:
      path = sorted(paths, key=os.path.getmtime)[-1]
      if len(paths) > 1:
        logging.warning('Found %d coredumps, taking most recent one %s',
                        len(paths), path)
      return path
    else:
      return None

  def _backtrace(self, core_file):
    """Print stack trace from core_file.

    Args:
      core_file: path to core file
    """
    echo_divider = r'echo {}\n'.format('-' * 80)
    # pyformat: disable
    command = ['gdb', 'binary', '-batch',
               '-ex', echo_divider,
               '-ex', 'echo bt full:\n',
               '-ex', 'bt full',
               '-ex', echo_divider,
               '-ex', 'echo thread apply all bt:\n',
               '-ex', 'thread apply all bt',
               '-ex', echo_divider,
               core_file]
    # pyformat: enable
    try:
      # Use gdb to print stack trace. Set a timeout to prevent a hung
      # gdb from hanging the entire cloudml training job.
      gdb_output = subprocess.check_output(command, timeout=300)
    except subprocess.CalledProcessError as e:
      # gdb segfaults when printing some stack traces. We take
      # whatever output was available prior to the crash. We do not
      # clean up any core dumps from gdb's crash itself, however,
      # since we will not copy them.
      gdb_output = e.output if e.output else None
    # If we had at least some stack trace, we log it. Otherwise, we
    # silently ignore the gdb failure.
    if gdb_output:
      logging.warning('%s\n%s\n%s', '=' * 80, gdb_output, '=' * 80)


class LogPipe(threading.Thread):
  """A file handle like object for sending data to a logger.

  This class can be used wherever a file handle is expected to pipe data
  to a logger.
  """

  def __init__(self, default_level, parse_logs=False):
    """Setup the object with a logger and a loglevel and start the thread."""
    threading.Thread.__init__(self)
    self.daemon = False
    self.default_level = default_level
    self.parse_logs = parse_logs
    self.fd_read, self.fd_write = os.pipe()
    self.pipe_reader = os.fdopen(self.fd_read)
    self.done = threading.Event()
    self.start()

  def fileno(self):
    """Return the write file descriptor of the pipe.

    Returns:
      The write file descriptor.
    """
    return self.fd_write

  def _get_log_level(self, log_level_name):
    """Return the log level corresponding to the given single-character name.

    Args:
      log_level_name: The name of the log level.

    Returns:
      The log level enum.
    """
    if log_level_name == 'D':
      return logging.DEBUG
    elif log_level_name == 'W':
      return logging.WARNING
    elif log_level_name == 'E':
      return logging.ERROR
    elif log_level_name == 'F':
      return logging.CRITICAL
    else:  # log_level_name == 'I'
      return logging.INFO

  def _log(self, level, line, extra):
    """Write to the log in a way that can be easily mocked out for testing.

    Args:
      level: The log level.
      line: The log line.
      extra: Extra info.
    """
    logging.log(level, line, extra=extra)

  def run(self):
    """Run the thread, logging everything.
    """
    try:
      for line in iter(self.pipe_reader.readline, ''):
        line = line.strip('\n')
        if not line:
          continue
        level, extra = self.default_level, None
        if self.parse_logs:
          level, line, extra = self.parse_log_line(line)
        self._log(level, line, extra)

    finally:
      self.pipe_reader.close()
      self.done.set()

  def parse_log_line(self, line):
    # We only try to parse TF CPP Logs and not TF Python logs. The python logs
    # are intercepted and written straight to the log file in sitecustomize.py.
    match = TENSORFLOW_CPP_LOG_FORMAT.match(line)
    if match:
      return self.get_tf_log_components(match)

    match = GRPC_LOG_FORMAT.match(line)
    if match:
      return self.get_grpc_log_components(match)

    return self.default_level, line, None

  def get_tf_log_components(self, match):
    level = self._get_log_level(match.group(1))
    extra = {
        'original_pathname': match.group(2),
        'original_lineno': int(match.group(3)),
    }
    line = match.group(4)
    return level, line, extra

  def get_grpc_log_components(self, match):
    level = self._get_log_level(match.group(1))
    now = datetime.datetime.now()
    year = now.year
    month = int(match.group(2))
    if month == 12 and now.month == 1:
      # New year elapsed since the log was emitted.
      year -= 1
    timestamp_int = time.mktime(
        datetime.datetime(
            year,
            month,
            int(match.group(3)),  # day
            int(match.group(4)),  # hour
            int(match.group(5)),  # minute
            int(match.group(6))).timetuple())  # second
    timestamp_float = float(timestamp_int) + (float(match.group(7)) /
                                              (1000 * 1000 * 1000))
    extra = {
        'original_created': timestamp_float,
        'original_thread': int(match.group(8)),
        'original_pathname': match.group(9),
        'original_lineno': int(match.group(10)),
    }
    line = match.group(11)
    return level, line, extra

  def close(self):
    """Close the write end of the pipe.
    """
    os.close(self.fd_write)

  def wait(self):
    """Wait until run() has finished.
    """
    self.done.wait()


def get_tf_config(cluster, task, job, running_project_id):
  """Get the data for the $TF_CONFIG environment variable.

  Args:
    cluster: dict, the arguments defining the cluster
    task: dict, the arguments defining the task
    job: dict, the arguments defining the job
    running_project_id: (Optional) The project id in which this is running.

  Returns:
    dict, the data for the TF_CONFIG variable (after JSON serialization)
  """
  # Setup the environment variable to pass configuration to the user's module.
  return {
      'cluster': cluster,
      'task': dict(task, cloud=running_project_id),
      'job': job,
      'environment': 'cloud'
  }


def is_single_replica_job(cluster):
  return not ('worker' in cluster or 'ps' in cluster or 'evaluator' in cluster)


def is_ucaip_job(cluster):
  # uCAIP jobs have an endpoint in the format of '...-workerpoolx-...'
  # 'chief' is always set in uCAIP jobs.
  chief_endpoints = cluster.get('chief', None)
  return chief_endpoints and ('workerpool' in chief_endpoints[0])


class Runner(object):
  """The runner class handles setup and execution of the user's program."""

  def __init__(self, stdout=None, stderr=None):
    """Initialize the runner.

    Args:
      stdout: (Optional) file handle to redirect stdout to when invoking
        subprocesses.
      stderr: (Optional) file handle to redirect stderr to when invoking
        subprocesses.
    """
    self._stdout = stdout
    self._stderr = stderr
    self.project_id = None

  @staticmethod
  def allow_coredumps():
    """Allows the called subprocess to produce a coredump."""

    limit_in_bytes = resource.RLIM_INFINITY
    resource.setrlimit(resource.RLIMIT_CORE, (limit_in_bytes, limit_in_bytes))

  def _check_call(self,
                  command,
                  stdout=None,
                  stderr=None,
                  env=None,
                  core_dump_glob_patterns=None,
                  cwd=None):
    """Run a command in a subprocess, providing debugging info if requested.

    Args:
      command: List of arguments to pass to the subprocess.
      stdout: (Optional) file handle to redirect stdout to when invoking
        subprocesses.
      stderr: (Optional) file handle to redirect stderr to when invoking
        subprocesses.
      env: If not None, a dictionary to override the environment of the
        subprocess.
      core_dump_glob_patterns: If not None, should be list of glob patterns used
        to find core dumps. If a core dump a matching core dump is found, then
        show stack traces and save the core dump.
      cwd: If not None, the subprocess's current directory will be changed to
        cwd before it is executed.

    Raises:
      ValueError: If the command is empty.
      subprocess.CalledProcessError: If the process exits with a non-zero exit
      code.
    """
    if not command:
      raise ValueError('command is empty')

    logging.info('Running command: %s', ' '.join(command))
    # Remember when this job started. We don't want core dumps prior to
    # now. Such core dumps can happen, say with gsutil failures or pip
    # install failures.
    start_time = datetime.datetime.now()
    proc = subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        preexec_fn=self.allow_coredumps,
        env=env)
    proc.wait()

    if proc.returncode != 0:
      if core_dump_glob_patterns:
        _FailureHandler(cwd, self.project_id,
                        start_time).on_failure(core_dump_glob_patterns)
      raise subprocess.CalledProcessError(proc.returncode, command)

  def download_and_install_packages(self, packages):
    """Download and install the supplied packages.

    Args:
      packages: A list of GCS URIs to pip packages to install.

    Raises:
      Exception: If download fails.
    """
    # Copy down the user's code from GCS.
    for package in packages:
      logging.info('Downloading the package: %s', package)
      name = package.rsplit('/', 1)[-1]
      # gsutil outputs everything to stderr so we need to divert it to stdout.
      try:
        self._check_call(
            ['gsutil', '-q', 'cp', package, name],
            stdout=self._stdout,
            stderr=self._stderr)
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Retrying after gsutil exception %s.', e)
        time.sleep(10)
        self._check_call(
            ['gsutil', '-q', 'cp', package, name],
            stdout=self._stdout,
            stderr=self._stderr)

      logging.info('Installing the package: %s', package)
      # We run pip with the option --upgrade so that users can
      # override modules already installed in the Cloud ML worker just
      # by providing their own version of the package. The primary
      # expected use is to allow users to supply their own version of
      # TensorFlow. We also run with --user so that we will not need
      # to uninstall the system package if there is a conflict.
      # However, we don't want to force-reinstall dependencies if we already
      # have the requested version, so we first install the user package with
      # --no-deps.
      command = [
          pip_version, 'install', '--user', '--upgrade', '--force-reinstall',
          '--no-deps', name
      ]
      # Then we install it a second time including dependencies but without
      # --force-reinstall, as the root package was installed above, this will
      # be a no-op if the package has no unresolved dependencies.
      deps_comm = [pip_version, 'install', '--user', name]
      max_trials = 2
      for trial in range(1, max_trials + 1):
        try:
          self._check_call(command, stdout=self._stdout, stderr=self._stderr)
          self._check_call(deps_comm, stdout=self._stdout, stderr=self._stderr)
          break
        except subprocess.CalledProcessError as e:
          if trial == max_trials:
            raise
          else:
            logging.warning('Installation of package failed on try %d/%d: %s\n'
                            'Retrying ...', trial, max_trials, e)

  def run(self, args, running_project_id):
    """Invoke the specified module in a subprocess.

    Args:
      args: The args provided by the Cloud ML service.
      running_project_id: (Optional) The project id in which this is running.

    Raises:
       ValueError: If arguments are invalid.
    """
    module_name = args.job.get('module_name', None)
    if not module_name:
      module_name = args.job.get('python_module', None)
    if not module_name:
      raise ValueError('A python_module must be specified.')
    logging.info('Running module %s.', module_name)

    packages = args.job.get('trainer_uri', [])
    packages.extend(args.job.get('package_uris', []))
    if packages:
      self.download_and_install_packages(packages)
    else:
      logging.info('No packages to install.')

    # Setup the list of arguments to pass to the user's module.
    job_args = args.job.get('job_args', [])
    if not job_args:
      job_args = args.job.get('args', [])
    if hasattr(args, 'hyperparams') and args.hyperparams:
      for k, v in iteritems(args.hyperparams):
        job_args.append('--' + k)
        job_args.append(v)
    job_dir = args.job.get('job_dir', None)
    if job_dir:
      job_args.append('--job-dir')
      trial = args.task.get('trial', None)
      if trial:
        job_args.append(os.path.join(job_dir, str(trial)))
      else:
        job_args.append(job_dir)

    env = dict(os.environ)

    # Don't set TF_CONFIG for single replica ucaip jobs.
    if not (is_ucaip_job(args.cluster) and is_single_replica_job(args.cluster)):
      config = get_tf_config(args.cluster, args.task, args.job,
                             running_project_id)
      env = dict(env, TF_CONFIG=json.dumps(config))

    if not (is_ucaip_job(args.cluster)) and hasattr(
        args, 'tpu_node') and args.tpu_node:
      self.wait_for_tpu_tf_server(args.tpu_node, TPU_JOB_RETRY_COUNT)
      env = dict(
          env, KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS=args.tpu_node['tpu_node_name'])

    # MKL variables, consistent with
    # https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu
    env = dict(env,
               KMP_BLOCKTIME='0',
               KMP_AFFINITY='granularity=fine,verbose,compact,1,0',
               KMP_SETTINGS='1',
               OMP_NUM_THREADS=str(multiprocessing.cpu_count()))

    command = [python_version, '-m', module_name] + job_args
    self._check_call(
        command,
        env=env,
        stdout=self._stdout,
        stderr=self._stderr,
        core_dump_glob_patterns=['core', 'core.python.*'])

  def force_cred_refresh(self):
    """Force to refresh the credential.

    Returns:
      The project id.
    """
    h = httplib2.Http()
    # Each of these tries can take ~30 seconds, so 6 tries could be up to 3
    # minutes.  If the metadata server is up, the first one should succeed.
    num_retries = 6
    for i in range(num_retries):
      try:
        GoogleCredentials.get_application_default().refresh(h)
        get_project = [
            'gcloud', 'config', 'list', 'project',
            '--format=value(core.project)'
        ]
        with open(os.devnull, 'w') as dev_null:
          project_id = subprocess.check_output(
              get_project, stderr=dev_null).strip()
          project_id = str(project_id, encoding='UTF-8')
          # Run gsutil to verify the auth can be set correctly.
          return_code = subprocess.call(['gsutil'],
                                        stdout=dev_null,
                                        stderr=dev_null)
          if return_code:
            raise OSError('Unable to initialize gsutil')
        if not project_id:
          raise ValueError('Unable to determine project')
        return project_id
      except Exception as e:  # pylint: disable=broad-except
        if i < num_retries - 1:
          logging.info('Retrying credential refresh after error %s.', e)
        else:
          # Preserves the traceback.
          reraise(CredentialsRefreshError, CredentialsRefreshError(e),
                  sys.exc_info()[2])
        time.sleep(10)

  def _run_tensorflow(self):
    """Run matrix multiplication on TensorFlow.
    """
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    n = 16
    dtype = tf.float32

    # TODO(b/158017886): remove workaround for TFE versioning once complete.
    if (parse_version(tf.__version__) < parse_version('2.0.0-alpha0') and
        'dlenv' not in tf.__version__):
      with tf.device('/gpu:0'):
        matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
        matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
        product = tf.matmul(matrix1, matrix2)

      # Avoid optimizing away redundant nodes and use soft device placement
      # in case of CPU training.
      config = tf.ConfigProto(
          graph_options=tf.GraphOptions(
              optimizer_options=tf.OptimizerOptions(
                  opt_level=tf.OptimizerOptions.L0)),
          allow_soft_placement=True)
      with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(product)
    else:
      # TF 2.0 did some breaking change. Use v1 compatible methods.
      # eager execution was enabled by default in TF 2.0
      # Disable eager execution to allow soft device placement in case of
      # CPU training.
      tf.compat.v1.disable_eager_execution()
      with tf.compat.v1.device('/gpu:0'):
        matrix1 = tf.compat.v1.Variable(tf.ones((n, n), dtype=dtype))
        matrix2 = tf.compat.v1.Variable(tf.ones((n, n), dtype=dtype))
        product = tf.compat.v1.matmul(matrix1, matrix2)

      # Avoid optimizing away redundant nodes and use soft device placement
      # in case of CPU training.
      config = tf.compat.v1.ConfigProto(
          graph_options=tf.compat.v1.GraphOptions(
              optimizer_options=tf.compat.v1.OptimizerOptions(
                  opt_level=tf.compat.v1.OptimizerOptions.L0)),
          allow_soft_placement=True)
      with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(product)

  def test_cuda_init(self):
    """Run test code in another process to release GPU memory completely.
    """

    try:
      multiprocessing.log_to_stderr()
      p = multiprocessing.Process(target=self._run_tensorflow)
      p.start()
      p.join()
      if p.exitcode != 0:
        raise Exception('Attempt of running TensorFlow on GPU failed!')
    except Exception as e:  # pylint: disable=broad-except
      reraise(CudaInitError, CudaInitError(e), sys.exc_info()[2])

  def get_tpu_cluster_resolver(self, tpu_node):
    """Return a TPUClusterResolver for the corresponding TF version.

    Args:
      tpu_node: dict that contains info about tpu node.

    Returns:
      TPUClusterResolver instance with information about the TPU.

    Raises:
      ValueError: If no TPUs are specified.
    """
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    # TODO(b/158017886): remove workaround for TFE versioning once complete.
    if (parse_version(tf.__version__) < parse_version('2.0.0-alpha0') and
        'dlenv' not in tf.__version__):
      return tf.contrib.cluster_resolver.TPUClusterResolver(
          tpu=[tpu_node['tpu_node_name']],
          zone=tpu_node['zone'],
          project=tpu_node['project'],
          job_name='worker')
    else:
      return tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu=[tpu_node['tpu_node_name']],
          zone=tpu_node['zone'],
          project=tpu_node['project'],
          job_name='worker')

  def wait_for_tpu_tf_server(self, tpu_node, num_retries):
    """Waits for TPU server.

    Args:
      tpu_node: dict that contains info about tpu node.
      num_retries: the number of retries in case of error

    Raises:
      ValueError: If fails to resolve TPU TF server.
    """
    logging.info('Checking the provisioning of TPU VM instance.')
    for i in range(num_retries):
      try:
        cluster_resolver = self.get_tpu_cluster_resolver(tpu_node)
        if 'worker' in cluster_resolver.cluster_spec().as_dict():
          break
      except Exception as e:  # pylint: disable=broad-except
        if i < num_retries - 1:
          logging.info('Still waiting for provisioning of TPU VM instance.')
        else:
          # Preserves the traceback.
          reraise(TpuIpResolveError, TpuIpResolveError(e), sys.exc_info()[2])
      time.sleep(10)


_EXITING_BECAUSE_TERMINATED_BY_SERVICE = False
_EXITING_BECAUSE_TERMINATED_BY_SERVICE_LOCK = threading.Lock()


def log_signal_and_quit(signum, frame):  # pylint: disable=unused-argument
  """Function handler for use with signal.signal().

  Logs which signal was caught and then terminates the process. Currently only
  used with SIGTERM.

  Args:
    signum: The single number.
    frame: Current stack frame. It is ignored.
  """

  message = ('Terminated by service. If the job is supposed to continue'
             ' running, it will be restarted on other VM shortly.')
  logging.info(message)
  with _EXITING_BECAUSE_TERMINATED_BY_SERVICE_LOCK:
    global _EXITING_BECAUSE_TERMINATED_BY_SERVICE
    _EXITING_BECAUSE_TERMINATED_BY_SERVICE = True
  # We follow the standard unix convention of setting the exit code to be
  # -signum. This is required so that the backend can determine whether
  # the process exited because it was killed by the system or if the user's
  # code exited with an error.
  sys.exit(-1 * signum)


# TODO(b/136507425): consider reuse below logic
def truncate_traceback(full_traceback, max_size_bytes):
  """Returns the traceback which is guaranteed to be within the size limit."""
  size = len(full_traceback)
  lines = full_traceback.split('\n')

  # If it is not of expected format, just keep the first max_size_bytes
  if lines[0] != DEFAULT_TRACEBACK_STARTING_LINE:
    return full_traceback[:max_size_bytes]

  # Assuming the traceback has the following structure:
  #
  # Traceback (most recent calls last):
  # <line number 1>
  # <code snippet 1>
  # <line number 2>
  # <code snippet 2>
  # ...
  # <the most recent line number>
  # <the most recent code snippet>
  # <lines with exception information>
  #
  # This means we can remove a few lines related to least recent calls.
  while size > max_size_bytes:
    if len(lines) < 4:
      return NO_TRACEBACK_MESSAGE
    # Delete the truncation sign if it's already inserted.
    if lines[1] == TRACEBACK_TRUNCATION_SIGN:
      del lines[1]
      size -= len(TRACEBACK_TRUNCATION_SIGN) + 1  # Include '\n'
    # Delete two lines related to the least recent call.
    size -= len(lines[1]) + len(lines[2]) + 2  # Include '\n'
    del lines[1]
    del lines[1]
    # Add a truncation sign.
    lines.insert(1, TRACEBACK_TRUNCATION_SIGN)
    size += len(TRACEBACK_TRUNCATION_SIGN) + 1  # Include '\n'
  return '\n'.join(lines)


def handle_exit_status(exit_code, exit_reason):
  """Function handler for recording exit status of the main function.

  Writes the exit code into termination log file and records the
  user error(with exit_code > 0) in the sentinel file to indicate it's
  non-retryable.

  Args:
    exit_code: The exit code of the main function.
    exit_reason: The cause of the exit.
  """

  # Creates the termination log message to be picked up by the Kubelet.
  # Gets the path from env variable TERMINATION_LOG, if it's not set, use
  # /dev/termination-log as the default path.
  termination_log_file_path = os.getenv('TERMINATION_LOG',
                                        DEFAULT_TERMINATION_LOG_PATH)

  traceback_file_path = os.environ.get('TRACEBACK_FILE_PATH',
                                       TRACEBACK_FILE_PATH)
  termination_log_content = {'exit_code': exit_code}
  termination_log_content['exit_reason'] = exit_reason
  if os.path.exists(traceback_file_path):
    termination_log_content['traceback'] = truncate_traceback(
        open(traceback_file_path, 'r').read(), MAX_TRACEBACK_SIZE_BYTES)

  with open(termination_log_file_path, 'w') as termination_log_file:
    termination_log_file.write(json.dumps(termination_log_content))
    if exit_code > 0 and exit_code <= 127:
      # The process exited with a non retryable error. We create a sentinel
      # file that will be used on container restart to detect that a previous
      # attempt failed with a permanent error and we should avoid retrying.
      # We can't use the TERMINATION_LOG because the ERROR_SENTINEL needs to
      # persist across container restarts. We don't want the TERMINATION_LOG
      # to persist across container restarts because we want each container
      # start to have its own unique log.
      #
      # If env variable ERROR_SENTINEL is not set, tries to use
      # dirname(LOG_FILE_TO_WRITE)/failed.sentinel, and if LOG_FILE_TO_WRITE
      # is not set, uses default value /var/log-storage/failed.sentinel.
      error_sentinel_file_path = os.getenv(
          'ERROR_SENTINEL',
          os.path.join(
              os.path.dirname(
                  os.getenv('LOG_FILE_TO_WRITE', DEFAULT_LOG_FILE_TO_WRITE)),
              DEFAULT_ERROR_SENTINEL_FILENAME))
      with open(error_sentinel_file_path, 'w') as error_sentinel_file:
        error_sentinel_file.write('Permanent error occurred.')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cluster', type=json.loads)
  parser.add_argument('--task', type=json.loads)
  parser.add_argument('--job', type=json.loads)
  parser.add_argument('--hyperparams', type=json.loads)
  parser.add_argument('--tpu_node', type=json.loads)
  args = parser.parse_args()

  log_wrap_info = LogPipe(logging.INFO, parse_logs=False)
  log_wrap_error = LogPipe(logging.ERROR, parse_logs=True)

  os.mkdir('user_dir')
  os.chdir('user_dir')

  # Sets a default of -1 that is interpreted as a retryable error.
  exit_code = -1
  exit_reason = 'UNKNOWN_EXIT_REASON'
  try:
    runner = Runner(stdout=log_wrap_info, stderr=log_wrap_error)
    runner.test_cuda_init()
    running_project_id = runner.force_cred_refresh()
    runner.project_id = running_project_id
    runner.run(args, running_project_id)
    exit_code = 0
    exit_reason = 'SUCCEEDED'
  except subprocess.CalledProcessError as e:
    logging.error(e)
    traceback.print_exc()
    exit_reason = 'SUBPROCESS_EXCEPTION'

    if e.returncode == -signal.SIGTERM:
      # For subprocess caused by SIGTERM, set termination log exit_code to
      # negative value to retry.
      exit_code = e.returncode
    else:
      # TODO(wenzhel): Temporarily set exit_code written to termination log
      # to abs(returncode) to identify all subprocess exceptions(except SIGTERM)
      # as user error. May loosen the criteria after getting better
      # understanding of the subprocess behavior (e.g. subprocess terminated by
      # other sys signals).
      # For runcloudml.py, still exit with original subprocess exception
      # returncode, this returncode will indicate the real cause of the
      # exception. e.g. 1 means error in user code, -15 means SIGTERM.
      exit_code = abs(e.returncode)
    sys.exit(abs(e.returncode))
  except CredentialsRefreshError as e:
    # Credential refresh error shouldn't be interpreted as permanent user
    # error, so exit with negative value of SIGUSER1.
    traceback.print_exc()
    logging.error(
        'Module raised an exception for failing to refresh credentials: %s.', e)
    exit_code = -1 * signal.SIGUSR1
    exit_reason = 'CREDENTIAL_REFRESH_EXCEPTION'
    sys.exit(exit_code)
  except TpuIpResolveError as e:
    # Treated as internal error.
    traceback.print_exc()
    logging.error('Module raised an exception for failing to TPU node IP: %s.',
                  e)
    exit_code = -1 * signal.SIGUSR1
    exit_reason = 'RESOLVE_TPU_IP_EXCEPTION'
    sys.exit(exit_code)
  except CudaInitError as e:
    # Treated as internal error.
    traceback.print_exc()
    logging.error('Module raised an exception for CUDA INIT failure: %s.', e)
    exit_code = -1 * signal.SIGUSR1
    exit_reason = 'CUDA_INIT_ERROR'
    sys.exit(exit_code)
  except Exception as e:  # pylint: disable=broad-except
    logging.error('Module raised an exception %s.', e)
    # reraise the exception.
    # We want to exit with a non-zero exit code in the event of failure
    # because the exit code is used to determine whether the job completed
    # successfully or not.
    exit_code = 1
    exit_reason = 'OTHER_EXCEPTION'
    raise
  except:  # pylint: disable=broad-except
    value_type, value = sys.exc_info()[:2]
    if isinstance(value, SystemExit) and value.code == -signal.SIGTERM:
      # Now, we only trap SIGTERM and convert it to a SystemExit. We should
      # handle it as retryable error.
      exit_code = value.code
    else:
      logging.error('Module raised an exception %s:%s.', value_type, value)
      exit_code = 1
    exit_reason = 'OTHER_ERROR'
    raise
  finally:
    # Writes the informative exit code to termination_log_file_path.
    handle_exit_status(exit_code, exit_reason)
    # Close the pipes so we flush the data.
    logging.info('Module completed; cleaning up.')
    log_wrap_info.close()
    log_wrap_error.close()
    logging.info('Clean up finished.')

  logging.info('Task completed successfully.')


if __name__ == '__main__':
  # Trap the SIGTERM signal for better debugging of logs
  signal.signal(signal.SIGTERM, log_signal_and_quit)

  logging.info('Running task with arguments: %s', ' '.join(sys.argv[1:]))
  main()