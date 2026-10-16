using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

// The launcher alone holds the job handle. Windows closes it even when the
// console is closed or the launcher is forcibly terminated, killing all workers.
public sealed class ServiceProcessScope : IDisposable
{
    private IntPtr job;
    public ServiceProcessScope()
    {
        job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception();
        var limits = new ExtendedLimits();
        limits.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(limits)))
        {
            int error = Marshal.GetLastWin32Error();
            Dispose();
            throw new Win32Exception(error);
        }
    }

    public Process Start(string executable, string arguments, string directory, string stdout, string stderr)
    {
        if (job == IntPtr.Zero) throw new ObjectDisposedException("ServiceProcessScope");
        var security = new SecurityAttributes();
        security.Length = Marshal.SizeOf(security);
        security.Inherit = true;
        IntPtr output = IntPtr.Zero, error = IntPtr.Zero, input = IntPtr.Zero;
        var info = new ProcessInformation();
        try
        {
            output = OpenFile(stdout, 0x40000000, 2, ref security);
            error = OpenFile(stderr, 0x40000000, 2, ref security);
            input = OpenFile("NUL", 0x80000000, 3, ref security);
            var startup = new StartupInfo();
            startup.Size = Marshal.SizeOf(startup);
            startup.Flags = 0x100; // STARTF_USESTDHANDLES
            startup.StdOutput = output;
            startup.StdError = error;
            startup.StdInput = input;
            // Suspend before any worker code runs, so no child can escape the job.
            if (!CreateProcess(executable, new StringBuilder("\"" + executable + "\" " + arguments),
                IntPtr.Zero, IntPtr.Zero, true, 0x08000004, IntPtr.Zero, directory, ref startup, out info))
                throw new Win32Exception();
            if (!AssignProcessToJobObject(job, info.Process))
                throw new Win32Exception();
            Process process = Process.GetProcessById((int)info.ProcessId);
            if (ResumeThread(info.Thread) == uint.MaxValue) throw new Win32Exception();
            return process;
        }
        catch
        {
            if (info.Process != IntPtr.Zero) TerminateProcess(info.Process, 1);
            throw;
        }
        finally
        {
            Close(input); Close(output); Close(error);
            Close(info.Thread); Close(info.Process);
        }
    }

    private static IntPtr OpenFile(string path, uint access, uint disposition, ref SecurityAttributes security)
    {
        IntPtr handle = CreateFile(path, access, 3, ref security, disposition, 0x80, IntPtr.Zero);
        if (handle == new IntPtr(-1)) throw new Win32Exception();
        return handle;
    }
    private static void Close(IntPtr handle)
    {
        if (handle != IntPtr.Zero && handle != new IntPtr(-1)) CloseHandle(handle);
    }
    public void Dispose()
    {
        if (job != IntPtr.Zero) { CloseHandle(job); job = IntPtr.Zero; }
    }

    [StructLayout(LayoutKind.Sequential)] private struct BasicLimits
    {
        public long ProcessTime, JobTime;
        public uint LimitFlags;
        public UIntPtr MinWorkingSet, MaxWorkingSet;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint Priority, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters
    {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] private struct ExtendedLimits
    {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [StructLayout(LayoutKind.Sequential)] private struct SecurityAttributes
    {
        public int Length;
        public IntPtr Descriptor;
        [MarshalAs(UnmanagedType.Bool)] public bool Inherit;
    }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] private struct StartupInfo
    {
        public int Size;
        public string Reserved, Desktop, Title;
        public uint X, Y, Width, Height, CountX, CountY, Fill, Flags;
        public ushort ShowWindow, ReservedSize;
        public IntPtr ReservedData, StdInput, StdOutput, StdError;
    }
    [StructLayout(LayoutKind.Sequential)] private struct ProcessInformation
    {
        public IntPtr Process, Thread;
        public uint ProcessId, ThreadId;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int type, ref ExtendedLimits limits, uint length);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes,
        IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string directory,
        ref StartupInfo startup, out ProcessInformation info);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateFile(string path, uint access, uint share, ref SecurityAttributes attributes,
        uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll")] private static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
}
